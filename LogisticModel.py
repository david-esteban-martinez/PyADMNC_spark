import math
import random
import sys

import numpy as np

from pyspark.ml.feature import OneHotEncoder

from pyspark.sql import DataFrame


def one_hot_encode(element, max_values):
    one_hot_encoded = []
    for i, value in enumerate(element):
        attribute_one_hot = [0] * (max_values[i])
        attribute_one_hot[int(value)] = 1
        one_hot_encoded.extend(attribute_one_hot)
    return one_hot_encoded


def _map_func(e, V, weights, lambda_num, n_continuous, n_features, numDiscrete, max_values):

    elemCont = e[-n_continuous:]
    elemCat = e[1:n_features - n_continuous + 1]
    elemCat = list(map(float, elemCat))
    elemCont = list(map(float, elemCont))

    elemCat = one_hot_encode(elemCat, max_values)
    index = random.randrange(numDiscrete)
    mask = np.zeros(numDiscrete)
    mask[index] = 1
    z = elemCat[index] if elemCat[index] == 1.0 else -1.0
    # Concatenate e.cPart and 1.0
    xCont = np.concatenate((elemCont, np.array([1.0])))
    # Assign yDisc to elem.mPart
    yDisc = mask
    # Calculate w
    first = np.matmul(V, xCont)
    second = np.matmul(weights, yDisc)
    w = np.dot(first, second)

    s = 1.0 / (1.0 + math.exp(z * w / lambda_num))  # TODO a veces w es enorme y crashea
    # Return a tuple of two arrays
    a1 = -s * z * (yDisc.transpose() * first.reshape((first.shape[0], 1)))
    a2 = -s * z * (xCont.transpose() * second.reshape((second.shape[0], 1)))

    return a1, a2

def getProbabilityEstimator(e, n_continuous, n_features, max_values,weights,V,lambda_num):
    elemCont = e[-n_continuous:]
    elemCat = e[1:n_features - n_continuous + 1]
    elemCat = list(map(float, elemCat))
    elemCont = list(map(float, elemCont))

    elemCat = one_hot_encode(elemCat, max_values)
    x = len(elemCat)
    wArray = np.array(weights, dtype="float")
    array = np.zeros(x)
    for i in range(x):
        yDisc = [0] * x
        yDisc[i] = 1

        xCont = np.append(elemCont, [1])

        if elemCat[i] == 1:
            z = 1
        else:
            z = -1
        first = np.matmul(V, xCont)
        second = np.matmul(wArray, yDisc)
        w = np.dot(first, second)
        p = 1.0 / (1.0 + math.exp(-z * w / lambda_num))
        array[i] = p
    return np.multiply.reduce(array)

UNCONVERGED_GRADIENTS = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max,
                         sys.float_info.max]
FULLY_CONVERGED_GRADIENTS = [sys.float_info.min, sys.float_info.min, sys.float_info.min, sys.float_info.min,
                             sys.float_info.min]





class LogisticModel:
    # TODO hacer la separación de parte numerica y continua automaticamente o con una opción si lo tienes precomputado
    def __init__(self, data: DataFrame, numContinuous, element, subspaceDimension, normalizing_radius, lambda_num,
                 spark):

        self.max_values = []
        self.enc = OneHotEncoder()
        self.n_features = len(element) - 1
        self.n_continuous = numContinuous
        self.lambda_num = lambda_num
        self.normalizing_radius = normalizing_radius

        self.regParameterScaleV = 1 / math.sqrt((numContinuous + 1) * subspaceDimension)
        self.lastGradientLogs = UNCONVERGED_GRADIENTS
        self.spark = spark
        numerical_columns = []
        for i in range(self.n_features - self.n_continuous, self.n_features):
            numerical_columns.append(str(i))
        categorical_columns = []
        for i in range(self.n_features - self.n_continuous):
            categorical_columns.append(str(i))
        X1 = data.drop(*numerical_columns)
        X1 = X1.drop("label")

        inputColumns=[]
        outputColumns=[]
        for i,column in enumerate(categorical_columns):
            inputColumns.append(str(i))
            outputColumns.append(str(i)+"_")
            X1 = X1.withColumn(column, X1[column].cast("float"))


        self.enc.setInputCols(inputColumns)
        self.enc.setOutputCols(outputColumns)
        self.enc.setHandleInvalid('keep')
        self.enc = self.enc.fit(X1)

        self.X1 = self.enc.transform(X1)
        self.X1 = self.X1.drop(*categorical_columns)
        self.numDiscrete = 0
        element = self.X1.first()
        for col in element:
            self.numDiscrete += len(col)
            self.max_values.append(len(col))

        self.V = np.ones((subspaceDimension, self.n_continuous + 1)) - 0.5
        self.weights = np.ones((subspaceDimension, self.numDiscrete)) - 0.5
        self.regParameterScaleW = 1 / math.sqrt(self.numDiscrete * subspaceDimension)


    def getProbabilityEstimator(self, element):
        V = self.V
        weights = self.weights
        lambda_num = self.lambda_num
        n_continuous = self.n_continuous
        n_features = self.n_features
        numDiscrete = self.numDiscrete
        max_values = self.max_values
        return getProbabilityEstimator(element,n_continuous,n_features,max_values,weights,V,lambda_num)


    def update(self, gradientW, gradientV, learningRate, regularizationParameter):
        # TODO falta normalizar, quizá hace falta
        self.weights = self.weights - learningRate * (
                gradientW + 2 * regularizationParameter * self.regParameterScaleW * self.weights)
        self.V = self.V - learningRate * (gradientV + 2 * regularizationParameter * self.regParameterScaleV * self.V)

        self.weights = self._normalizeColumns(self.weights)
        self.V = self._normalizeColumns(self.V)

    def _normalizeColumns(self, m):
        for i in range(m.shape[1]):
            total = 0.0
            for j in range(m.shape[0]):
                total += m[j][i] * m[j][i]
            total = math.sqrt(total)
            if total > self.normalizing_radius:
                for j in range(m.shape[0]):
                    m[j][i] = m[j][i] * self.normalizing_radius / total
        return m

    def one_hot_encode(self, element):
        one_hot_encoded = []
        for i, value in enumerate(element):
            attribute_one_hot = [0] * (self.max_values[i])
            attribute_one_hot[int(value)] = 1
            one_hot_encoded.extend(attribute_one_hot)
        return one_hot_encoded


    def _reduce_func(self, e1, e2):
        return e1[0] + e2[0], e1[1] + e2[1]

    def trainWithSGD(self, data: DataFrame, maxIterations, minibatchFraction,
                     regParameter, learningRate0, learningRateSpeed):

        # SGD
        consecutiveNoProgressSteps = 0
        i = 1

        while ((i < maxIterations) and (consecutiveNoProgressSteps < 10)):
            # Finishing condition: gradients are small for several iterations in a row
            # Use a minibatch instead of the whole dataset
            minibatch = data.sample(False, minibatchFraction)

            total = minibatch.count()  # TODO la versión de Scala hace un minibatch de tamaño aproximado, algo random
            while total <= 0:
                minibatch = data.sample(False, minibatchFraction)
                total = minibatch.count()
            learningRate = learningRate0 / (1 + learningRateSpeed * (i - 1))


            V = self.V
            weights = self.weights
            lambda_num = self.lambda_num
            n_continuous = self.n_continuous
            n_features = self.n_features
            numDiscrete = self.numDiscrete
            max_values = self.max_values
            (sumW, sumV) = minibatch.rdd.map(
                lambda e: _map_func(e, V, weights, lambda_num, n_continuous, n_features, numDiscrete, max_values)) \
                .reduce(lambda a,b: (a[0] + b[0], a[1] + b[1]))

            gradientW = sumW / total

            gradientProgress = sum(map(math.fabs, gradientW.flatten()))

            if (gradientProgress < 0.00001):
                consecutiveNoProgressSteps = consecutiveNoProgressSteps + 1
            else:
                consecutiveNoProgressSteps = 0

            # DEBUG
            # if (i % 10 == 0)
            #     println("Gradient size:" + gradientProgress)

            # println("Gradient size:"+gradientProgress)

            if (i >= maxIterations - len(self.lastGradientLogs)):
                self.lastGradientLogs[i - maxIterations + len(self.lastGradientLogs)] = math.log(gradientProgress)

            self.update(gradientW, (sumV / total), learningRate, regParameter)

            i = i + 1
        if (consecutiveNoProgressSteps >= 10):
            self.lastGradientLogs = FULLY_CONVERGED_GRADIENTS

    def getProbabilityEstimators(self, elements):
        return list(map(self.getProbabilityEstimator, elements))
