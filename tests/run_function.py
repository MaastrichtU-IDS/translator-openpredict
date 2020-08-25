from joblib import load
from openpredict.openpredict_api import get_predict
from openpredict.openpredict_omim_drugbank import get_drug_disease_classifier
from datetime import datetime

time_start = datetime.now()
clf, scores = get_drug_disease_classifier()

## Run it:
# python3 tests/run_function.py

# Call get predict from API for a DRUG
prediction_result = get_predict('DRUGBANK:DB00394')

print('PredictRuntime: ' + str(datetime.now() - time_start))

print('Results:')
print(prediction_result)
