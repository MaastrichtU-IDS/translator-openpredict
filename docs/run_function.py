from joblib import load
from openpredict.openpredict_api import get_predict, init_openpredict_dir
from openpredict.openpredict_model import train_model
from openpredict.rdf_utils import add_feature_metadata, add_run_metadata, retrieve_features
from datetime import datetime
import pandas as pd

### Run it:
# python3 tests/run_function.py

### Test train + predict
# time_start = datetime.now()
# clf, scores, hyper_params = train_model()
# time_build = datetime.now()

# # Call get predict from API for a DRUG
# prediction_result = get_predict('DRUGBANK:DB00394')

# print('Build runtime: ' + str(time_build - time_start))
# print('Predict runtime: ' + str(datetime.now() - time_build))
# print('Total runtime: ' + str(datetime.now() - time_start))
# print('Results:')
# print(prediction_result)

### Test Spark:
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


### Print Dataframes
# clf = load('openpredict/data/models/openpredict-baseline-omim-drugbank.joblib') 
# print(clf.feature_names)

# (drug_df, disease_df) = load('openpredict/data/features/drug_disease_dataframes.joblib')

# drug_features_df = drug_df.columns.get_level_values(0).drop_duplicates()
# disease_features_df = disease_df.columns.get_level_values(0).drop_duplicates()

# print(drug_features_df)
# print(disease_features_df)

# print(drug_df.index)
# ## length=505 drugs

# print(disease_df.index)
# ## length=300

# print(drug_df.head())
# print(disease_df.head())
# print(drug_df.columns.names)

# # Get the features in the dataframe (column 1 is the drug ID)
# drug_features_df = drug_df.columns.get_level_values(0).drop_duplicates()
# # print(drug_df.columns.get_level_values(0).drop_duplicates(keep=False))
# # df.drop_duplicates(keep=False, inplace=True)

# print(drug_features_df)
# pd.set_option("display.max_rows", 10, "display.max_columns", None)
# print(drug_df)
    # (drug_df, disease_df)= load(pkg_resources.resource_filename('openpredict', 'data/features/drug_disease_dataframes.joblib'))
# print(features.feature_names)


### Generate RDF metadata for baseline features and first run

# add_feature_metadata("GO-SIM", "GO based drug-drug similarity", "Drugs")
# add_feature_metadata("TARGETSEQ-SIM", "Drug target sequence similarity: calculation of SmithWaterman sequence alignment scores", "Drugs")
# add_feature_metadata("PPI-SIM", "PPI based drug-drug similarity, calculate distance between drugs on protein-protein interaction network", "Drugs")
# add_feature_metadata("TC", "Drug fingerprint similarity, calculating MACS based fingerprint (substructure) similarity", "Drugs")
# add_feature_metadata("SE-SIM", "Drug side effect similarity, calculating Jaccard coefficient based on drug sideefects", "Drugs")
# add_feature_metadata("PHENO-SIM", "Disease Phenotype Similarity based on MESH terms similarity", "Diseases")
# add_feature_metadata("HPO-SIM", "HPO based disease-disease similarity", "Diseases")

# hyper_params = {
#     'penalty': 'l2',
#     'dual': False,
#     'tol': 0.0001,
#     'C': 1.0,
#     'random_state': 100
# }
# scores = {'precision': 0.8602150537634409, 'recall': 0.7228915662650602, 'accuracy': 0.8683417085427135, 
#     'roc_auc': 0.8988169874066402, 'f1': 0.7855973813420621, 'average_precision': 0.8733631857757298}

# model_features = retrieve_features('All').keys()

# add_run_metadata(scores, model_features, hyper_params)
