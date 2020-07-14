from joblib import load
from openpredict.feature_generation import generate_feature
from openpredict.openpredict_api import get_predict

## Run it:
# python3 tests/run_function.py

## Get prediction for Drug

entity = 'DB00394'

# Call get predict from API for a DRUG
prediction_result = get_predict('DB00394', 'drug', 'disease')

print(prediction_result)


## Feature generation
# generate_feature()