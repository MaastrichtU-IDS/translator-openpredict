from joblib import load
from openpredict.feature_generation import generate_feature

## Run it:
# python3 tests/run_function.py

## Get prediction for Drug

# entity = 'DB00570'

# clf = load('data/models/drug_disease_model.joblib') 

# prediction_result = clf.predict([[entity]]).reshape(1, 1)
# print(prediction_result)


## Feature generation
generate_feature()