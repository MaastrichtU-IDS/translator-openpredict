from joblib import load

entity = 'DB00570'

clf = load('data/models/drug_disease_model.joblib') 

prediction_result = clf.predict([[entity]]).reshape(1, 1)
print(prediction_result)