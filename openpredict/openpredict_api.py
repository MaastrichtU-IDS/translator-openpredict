import os
import pandas as pd
import numpy as np
import connexion
import logging
import json
from joblib import load
from openpredict.build_models import get_drug_disease_classifier
from openpredict.build_models import mergeFeatureMatrix
from openpredict.build_models import createFeatureDF
from sklearn.linear_model import LogisticRegression

def start_api(port=8808, debug=False):
    """Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

    :param port: Port of the OpenPredict API, defaults to 8808
    :param debug: Run in debug mode, defaults to False
    """
    print("Starting the \033[1mTranslator OpenPredict API\033[0m 🔮🐍")
    
    api = connexion.App(__name__, options={"swagger_url": ""})

    api.add_api('../openapi.yml', validate_responses=True)

    if debug:
        # Run in development mode
        deployment_server='flask'
        logging.basicConfig(level=logging.DEBUG)
        print("Development deployment using \033[1mFlask\033[0m 🧪")
        print("Debug enabled 🐞 - The API will reload automatically at each change 🔃")
    else:
        # Run in productiom with tornado (also available: gevent)
        deployment_server='tornado'
        logging.basicConfig(level=logging.INFO)
        print("Production deployment using \033[1mTornado\033[0m 🌪️")
    
    print("Access Swagger UI at \033[1mhttp://localhost:" + str(port) + "\033[1m 🔗")
    api.run(port=port, debug=debug, server=deployment_server)



### Code for the different calls of the app

def get_predict(entity, input_type, predict_type):
    """Get predicted associations for a given entity.
    
    :param entity: Search for predicted associations for this entity
    :param input_type: Type of the entity in the input (e.g. drug, disease)
    :param predict_type: Type of the predicted entity in the output (e.g. drug, disease)
    :return: Prediction results object with score
    """

    resources_folder = "data/resources/"
    features_folder = "data/features/"
    drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv', 
                     'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    drugfeatfiles = [ os.path.join(features_folder, fn) for fn in drugfeatfiles]
    diseasefeatfiles = [ os.path.join(features_folder, fn) for fn in diseasefeatfiles]

    ## Get all DFs
    # Merge feature matrix
    drug_df, disease_df = mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
    drugDiseaseKnown = pd.read_csv(resources_folder + 'openpredict-omim-drug.csv',delimiter=',') 
    drugDiseaseKnown.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)

    # TODO: save
    drugDiseaseDict  = set([tuple(x) for x in  drugDiseaseKnown[['Drug','Disease']].values])

    drugwithfeatures = set(drug_df.columns.levels[1].tolist())
    diseaseswithfeatures = set(disease_df.columns.levels[1].tolist())

    # TODO: save
    commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )

    # Load classifier
    clf = load('data/models/drug_disease_model.joblib') 


    pairs=[]
    classes=[]
    if input_type == "drug":
        # Input is a drug, we only iterate on disease
        dr = entity
        for di in commonDiseases:
            cls = (1 if (dr,di) in drugDiseaseDict else 0)
            pairs.append((dr,di))
            classes.append(cls)
    else: 
        # Input is a disease
        di = entity
        for dr in commonDrugs:
            cls = (1 if (dr,di) in drugDiseaseDict else 0)
            pairs.append((dr,di))
            classes.append(cls)
    classes = np.array(classes)
    pairs = np.array(pairs)
    test_df = createFeatureDF(pairs, classes, drugDiseaseKnown.values, drug_df, disease_df)

    # Get list of drug-disease pairs (should be saved somewhere from previous computer?)
    # Another API: given the type, what kind of entities exists?
    # Getting list of Drugs and Diseases:
    # commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    # commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )
    features = list(test_df.columns.difference(['Drug','Disease','Class']))
    y_proba = clf.predict_proba(test_df[features])
    prediction_df = pd.DataFrame( list(zip(pairs[:,0], pairs[:,1], y_proba[:,1])), columns =['Drug','Disease','score'])

    prediction_results=prediction_df.to_json(orient='records')
    prediction_json=json.loads(prediction_results)
    # print('Prediction RESULTS')
    # print(prediction_results)
    #prediction_results = {
    #    'results': [{'source' : entity, 'target': 'associated drug 1', 'score': 0.8}],
    #    'count': 1
    #}
    ## Currently returns:
    # "[{\"Drug\":\"DB00394\",\"Disease\":\"132300\",\"score\":0.0692499628},{\"Drug\":\"DB00394\",\"Disease\":\"145200\",\"score\":0.2462079817},{\"Drug\":\"DB00394\",\"Disease\":\"606798\",\"score\":0.0394063656}
    return {'results': prediction_json, 'count': len(prediction_results)} or ('Not found', 404)

# TODO: get_predict wrapped in ReasonerStdApi
def post_reasoner_predict(request_body):
    """Get predicted associations for a given ReasonerAPI query.
    
    :param request_body: The ReasonerStdAPI query in JSON
    :return: Predictions as a ReasonerStdAPI Message
    """
    prediction_result = {
        "query_graph": {
            "nodes": [
                {
                    "id": "n00",
                    "type": "Drug"
                },
                {
                    "id": "n01",
                    "type": "Disease"
                }
            ],
            "edges": [
                {
                    "id": "e00",
                    "type": "Association",
                    "source_id": "n00",
                    "target_id": "n01"
                }
            ]
        },
        "query_options": {
            "https://w3id.org/openpredict/prediction/score": "0.7"
        }
    }
    return prediction_result or ('Not found', 404)