
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
import math

import numpy as np

from sklearn.model_selection import  KFold

from LogisticModel import LogisticModel
import pyspark.sql.functions as f
import GMMclustering

import os

os.environ['SPARK_HOME'] = r'C:\Users\david\Spark'

os.environ['PYSPARK_PYTHON'] = 'python'


def create_spark_context(app_name: str) -> SparkContext:
    conf = (
        SparkConf()
        .set("spark.driver.memory", "8g")
        .set("spark.sql.session.timeZone", "UTC")
        .setMaster("local[4]")
        .setAppName(app_name)
    )

    spark_context = SparkContext(conf=conf)

    return spark_context


def create_spark_session(app_name: str) -> SparkSession:
    conf = (
        SparkConf()
        .set("spark.driver.memory", "8g")
        .set("spark.sql.session.timeZone", "UTC")
    )

    spark_session = SparkSession \
        .builder \
        .master("local[4]") \
        .config(conf=conf) \
        .appName(app_name) \
        .getOrCreate()

    return spark_session

def getProbabilityEstimator(element,gmm,V,weights,lambda_num, n_continuous, max_values,n_features):

    gmmEstimator = gmm.scoreOnePoint(np.array(element)[-n_continuous:])[0]
    logisticEstimator = getProbabilityEstimatorLogistic(element,n_continuous,n_features,max_values,weights,V,lambda_num)
    # DEBUG
    # print("gmm: " + str(gmmEstimator) + "   logistic: " + str(logisticEstimator))
    return element[0],math.log(logisticEstimator) * gmmEstimator  # TODO el score ya hace log, no hace falta log otra vez?
def getProbabilityEstimatorLogistic(e, n_continuous, n_features, max_values,weights,V,lambda_num):
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
def one_hot_encode(element, max_values):
    one_hot_encoded = []
    for i, value in enumerate(element):
        attribute_one_hot = [0] * (max_values[i])
        attribute_one_hot[int(value)] = 1
        one_hot_encoded.extend(attribute_one_hot)
    return one_hot_encoded
def callGMM(gmm,element):
    return gmm.transform(element)
def isAnomaly(element,gmm2,first_continuous,V,weights,lambda_num,n_continuous,max_values,n_features,threshold):

    return getProbabilityEstimator(element,gmm2,V,weights,lambda_num,n_continuous,max_values,n_features) < threshold

class ADMNC_LogisticModel:
    DEFAULT_SUBSPACE_DIMENSION = 10
    DEFAULT_REGULARIZATION_PARAMETER = 1.0
    DEFAULT_LEARNING_RATE_START = 1.0
    DEFAULT_LEARNING_RATE_SPEED = 0.1
    DEFAULT_FIRST_CONTINUOUS = 2
    DEFAULT_MINIBATCH_SIZE = 100
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_GAUSSIAN_COMPONENTS = 4
    DEFAULT_NORMALIZING_R = 10.0
    DEFAULT_LOGISTIC_LAMBDA = 1.0

    def __init__(self, subspace_dimension=DEFAULT_SUBSPACE_DIMENSION,
                 regularization_parameter=DEFAULT_REGULARIZATION_PARAMETER,
                 learning_rate_start=DEFAULT_LEARNING_RATE_START,
                 learning_rate_speed=DEFAULT_LEARNING_RATE_SPEED,
                 gaussian_num=DEFAULT_GAUSSIAN_COMPONENTS,
                 normalizing_radius=DEFAULT_NORMALIZING_R,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 logistic_lambda=DEFAULT_LOGISTIC_LAMBDA,
                 minibatch_size=DEFAULT_MINIBATCH_SIZE,
                 first_continuous=0, threshold=-1.0, anomaly_ratio=0.1):
        # TODO los atributos logistic y gmm solo son para que funcione BayesianSearch, hay que buscar otras alternativas
        # min y max lo mismo, y estimator, y classes

        self.gmm = None
        self.logistic = None
        self.first_continuous = first_continuous
        self.logistic_lambda = logistic_lambda
        self.max_iterations = max_iterations
        self.normalizing_radius = normalizing_radius
        self.gaussian_num = gaussian_num
        self.learning_rate_speed = learning_rate_speed
        self.learning_rate_start = learning_rate_start
        self.regularization_parameter = regularization_parameter
        self.subspace_dimension = subspace_dimension
        self.minibatch_size = minibatch_size

        self.threshold = threshold
        self.anomaly_ratio = anomaly_ratio

    def fit(self, ss: SparkSession, data, y=None):

        sampleElement = data.first()
        numElems = data.count()
        minibatchFraction = self.minibatch_size / numElems
        if minibatchFraction > 1:
            minibatchFraction = 1

        n_features = len(sampleElement)
        self.logistic = LogisticModel(data, n_features-1 - self.first_continuous,
                                      sampleElement,
                                      self.subspace_dimension, self.normalizing_radius, self.logistic_lambda,ss)
        self.logistic.trainWithSGD(data, self.max_iterations, minibatchFraction, self.regularization_parameter,
                                   self.learning_rate_start,
                                   self.learning_rate_speed)

        gmm2=GMMclustering.GMMclustering()
        categorical_columns = []
        numerical_columns = []
        for i in range(self.first_continuous, n_features - 1):
            numerical_columns.append(str(i))
        for i in range(0, self.first_continuous):
            categorical_columns.append(str(i))

        categorical_columns = list(map(lambda x: str(x), categorical_columns))

        columns = []
        for col in numerical_columns:
            columns.append(f.col(col))
        data_cont = data.drop(*categorical_columns)

        data_cont = data_cont.drop("label")

        gmm2.fit(data_cont.rdd.map(list),n_components=4)
        self.gmm=gmm2

        V = self.logistic.V
        weights = self.logistic.weights
        lambda_num = self.logistic.lambda_num
        n_continuous = self.logistic.n_continuous
        n_features = self.logistic.n_features
        numDiscrete = self.logistic.numDiscrete
        max_values = self.logistic.max_values
        estimators = data.rdd.map(lambda e:getProbabilityEstimator(e,gmm2,V,weights,lambda_num,n_continuous,max_values,n_features))

        targetSize = int(numElems * self.anomaly_ratio)
        if targetSize <= 0: targetSize = 1
        topValues=estimators.top(targetSize)#TODO, un valor más negativo es más anómalo o es al revés?
        self.threshold = topValues[targetSize-1][1]


    def getProbabilityEstimator(self, element):
        gmmEstimator = self.gmm.scoreOnePoint(np.array(element)[-self.logistic.n_continuous:])[0]
        logisticEstimator = self.logistic.getProbabilityEstimator(element)
        # DEBUG
        # print("gmm: " + str(gmmEstimator) + "   logistic: " + str(logisticEstimator))
        return math.log(logisticEstimator) * gmmEstimator

    def getProbabilityEstimators(self, elements):

        return elements.rdd.map(lambda e:self.getProbabilityEstimator(e))




    def isAnomaly(self, element,gmm2,V,weights,lambda_num,n_continuous,max_values,n_features,threshold):

        return getProbabilityEstimator(element,gmm2,V,weights,lambda_num,n_continuous,max_values,n_features) < threshold

    # set_params: a function that sets the parameters of the model
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # get_params: a function that returns the parameters of the model
    def get_params(self, deep=True):
        params = {}
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ADMNC_LogisticModel) and deep:
                params[key] = self.__dict__[key].get_params(deep)
            else:
                params[key] = self.__dict__[key]
        return params

    # get_param_names: a function that returns the names of the parameters of the model
    def get_param_names(self):
        names = []
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ADMNC_LogisticModel):
                names.extend(self.__dict__[key].get_param_names())
            else:
                names.append(key)
        return names

    def predict_proba(self, elements):
        results = np.zeros((elements.shape[0], 2))
        for i in range(len(elements)):
            result = self.getProbabilityEstimator(elements[i])
            interpolation = self.interpolate(result)
            results[i] = np.array([interpolation, 1 - interpolation])
        return results

    def findMinMax(self, data):
        results = self.getProbabilityEstimators(data)
        self.max = max(results)
        self.min = min(results)

    def interpolate(self, value):
        t = (value - self.min) / (self.max - self.min)
        return t

    def predict(self, elements):
        result = np.zeros((elements.shape[0]))
        for i in range(len(elements)):
            if self.getProbabilityEstimator(elements[i]) < self.threshold:
                result[i] = 1
            else:
                result[i] = 0
        return result

    def decision_function(self, elements):
        return self.getProbabilityEstimators(elements)


def kfold_csr(data, y, k, randomize=False, remove_ones=False):
    # data: una base de datos en formato csr_matrix
    # k: el número de pliegues para la validación cruzada
    # randomize: un booleano que indica si queremos mezclar los datos antes de dividirlos
    # remove_ones: un booleano que indica si queremos eliminar las filas con etiqueta 1 del conjunto de entrenamiento
    # Devuelve: una lista de tuplas, cada una con dos arrays de índices para el conjunto de entrenamiento y el de prueba

    data = data.toarray()

    labels = y

    kf = KFold(n_splits=k, shuffle=randomize)

    results = []

    for train_index, test_index in kf.split(data):
        if remove_ones:
            mask = labels[train_index] == 0
            train_index = train_index[mask]
        results.append((train_index, test_index))

    return results

#We had to make every function on the same file because Pyspark can't pass functions in an RDD, it gives Pickling error
if __name__ == '__main__':
    sc = create_spark_context("ADMNC")
    spark = create_spark_session("ADMNC")
    admnc = ADMNC_LogisticModel(first_continuous=13, subspace_dimension=2, logistic_lambda=0.1,
                                regularization_parameter=0.001, learning_rate_start=1,
                                learning_rate_speed=0.2, gaussian_num=2, normalizing_radius=10, anomaly_ratio=0.2)


    df = spark.read \
        .format("csv") \
        .option("numFeatures", 23) \
        .option("inferSchema", True) \
        .option("header", True) \
        .load(path="reduced_data_movie_0.03_4_0.4_0.3_random.csv")
    # df.show()


    admnc.fit(spark, df)
    logistic = admnc.logistic
    first_continuous = admnc.first_continuous
    gmm=admnc.gmm
    V = logistic.V
    weights = logistic.weights
    lambda_num = logistic.lambda_num
    n_continuous = logistic.n_continuous
    n_features = logistic.n_features
    numDiscrete = logistic.numDiscrete
    max_values = logistic.max_values
    threshold = admnc.threshold
    results = df.rdd.map(lambda e: isAnomaly(e,gmm,first_continuous,V,weights,lambda_num,n_continuous,max_values,n_features,threshold))

    estimators = df.rdd.map(
        lambda e: getProbabilityEstimator(e, gmm, V, weights, lambda_num, n_continuous, max_values,
                                          n_features))

    resultsDataframe=estimators.toDF(["label","estimator"])
    evaluator = BinaryClassificationEvaluator()
    evaluator.setRawPredictionCol("estimator")
    print(evaluator.evaluate(resultsDataframe))

