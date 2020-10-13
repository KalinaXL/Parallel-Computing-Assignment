import argparse
import os
from time import time
from contextlib import contextmanager
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

algorithms = ['GBTClassifier', 'LogisticRegression', 'RandomForestClassifier', 'LinearSVC']

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--algorithm', choices = algorithms, default = 0, type = int, help = 'index of algorithm in the list')
ap.add_argument('-n', '--app-name', default = 'pyspark_app', type = str, help = 'name of the app')
ap.add_argument('-f', '--num-features', default = 100000, type = int, help = '# of the features of hashing tf')
args = ap.parse_args()

@contextmanager
def timer():
  start = time()
  yield
  end = time()
  print(f'Elapsed time: {end - start: .4f}s')


algorithm = algorithms[args.algorithm]
spark = SparkSession.builder.appName(args.app_name).getOrCreate()

with timer():
  print('[INFO] Reading time')
  rdd = spark.sparkContext.textFile(os.path.join('dataset', 'train.ft.txt'))
  rdd.cache()

tokenizer = Tokenizer(inputCol = 'rawContent', outputCol = 'words')
hashing_tf = HashingTF(numFeatures = args.num_features, inputCol = 'words', outputCol = 'rawFeatures')
idf = IDF(inputCol = 'rawFeatures', outputCol = 'features')
label_indexer = StringIndexer(inputCol = 'rawLabel', outputCol = 'label')

with timer():
  print('[INFO] Preprocessing time')
  df = rdd.map(lambda x: (x[:10], x[11: ])).toDF(['rawLabel', 'rawContent'])
  df = tokenizer.transform(df)
  tf_df = hashing_tf.transform(df)
  # tf_df.cache()
  idf_model = idf.fit(tf_df)
  encoded_df = idf_model.transform(tf_df)
  training_df = encoded_df.select('rawLabel', 'features')
  labelModel = label_indexer.fit(training_df)
  training_df = labelModel.transform(training_df)

if algorithm in ['LogisticRegression', 'LinearSVC']:
  with timer():
    print('[INFO] Normalization time')
    sc = StandardScaler(inputCol = 'features', outputCol = 'scaled_features')
    sc_model = sc.fit(training_df)
    training_df = sc_model.transform(training_df)
    training_df = training_df.drop('scaled_features')
    training_df = training_df.withColumnRenamed('scaled_features', 'features')

with timer():
  print('[INFO] Training time')
  logistic_model = eval(algorithm)()
  classifier = logistic_model.fit(training_df)

with timer():
  print('[INFO] Inference time')
  preds = classifier.transform(training_df)

evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'prediction')

with timer():
  acc = evaluator.evaluate(preds)
  print(f'[RESULT] Accuracy of training set: {acc:.4f}')
  print('[INFO] Evaluation time')

with timer():
  rdd = spark.sparkContext.textFile(os.path.join('dataset', 'test.ft.txt'))
  rdd.cache()
  df = rdd.map(lambda x: (x[:10], x[11: ])).toDF(['rawLabel', 'rawContent'])
  df = tokenizer.transform(df)
  tf_df = hashing_tf.transform(df)
  # tf_df.cache()
  encoded_df = idf_model.transform(tf_df)
  test_df = encoded_df.select('rawLabel', 'features')
  test_df = labelModel.transform(test_df)
  if algorithm in ['LogisticRegression', 'LinearSVC']:
    test_df = sc_model.transform(test_df)
    test_df = test_df.drop('scaled_features')
    test_df = test_df.withColumnRenamed('scaled_features', 'features')
  preds = classifier.transform(test_df)
  acc = evaluator.evaluate(preds)
  print(f'[RESULT] Accuracy of test set: {acc:.4f}')
  print('[INFO] Evaluating test set')

