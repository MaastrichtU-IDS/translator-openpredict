from joblib import load
from openpredict.openpredict_api import get_predict

## Run it:
# python3 tests/run_function.py

# Call get predict from API for a DRUG
prediction_result = get_predict('DRUGBANK:DB00394')

print('Results:')
print(prediction_result)
