from joblib import load
from openpredict.openpredict_api import get_predict
from openpredict.openpredict_omim_drugbank import build_drug_disease_classifier
from openpredict.build_utils import generate_classifier_metadata
from datetime import datetime

## Run it:
# python3 tests/run_function.py

# time_start = datetime.now()
# clf, scores = build_drug_disease_classifier()

# time_build = datetime.now()

# # Call get predict from API for a DRUG
# prediction_result = get_predict('DRUGBANK:DB00394')

# print('Build runtime: ' + str(time_build - time_start))
# print('Predict runtime: ' + str(datetime.now() - time_build))
# print('Total runtime: ' + str(datetime.now() - time_start))

# print('Results:')
# print(prediction_result)

# import findspark
# from pyspark import SparkConf, SparkContext
# findspark.init()

# config = SparkConf()
# config.setMaster("local[*]")
# config.set("spark.executor.memory", "5g")
# config.set('spark.driver.memory', '5g')
# config.set("spark.memory.offHeap.enabled",True)
# config.set("spark.memory.offHeap.size","5g") 
# sc = SparkContext(conf=config, appName="OpenPredict")
# print (sc)


scores = {'precision': 0.8602150537634409, 'recall': 0.7228915662650602, 'accuracy': 0.8683417085427135, 
    'roc_auc': 0.8988169874066402, 'f1': 0.7855973813420621, 'average_precision': 0.8733631857757298}

generate_classifier_metadata('openpredict-omim-drugbank-0', scores, "Original OpenPredict classifier based on OMIM and DrugBank")