from joblib import load
from openpredict.openpredict_api import get_predict
from openpredict.openpredict_omim_drugbank import get_drug_disease_classifier
from datetime import datetime

## Run it:
# python3 tests/run_function.py

time_start = datetime.now()
clf, scores = get_drug_disease_classifier()

time_build = datetime.now()

# Call get predict from API for a DRUG
prediction_result = get_predict('DRUGBANK:DB00394')

print('Build runtime: ' + str(time_build - time_start))
print('Predict runtime: ' + str(datetime.now() - time_build))
print('Total runtime: ' + str(datetime.now() - time_start))

# print('Results:')
# print(prediction_result)
