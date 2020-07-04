import connexion
import logging
from openpredict.compute_similarities import get_drug_disease_similarities

def start_api(port=8808, debug=False):
    """Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

    :param port: Port of the OpenPredict API, defaults to 8808
    :param debug: Print debug logs, defaults to False
    """
    print("Starting the \033[1mTranslator OpenPredict API\033[0m 🔮🐍")
    
    api = connexion.App(__name__)
    # app = connexion.FlaskApp(__name__, server='gevent')
    # app = connexion.App(__name__, server='tornado')

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
    
    print("Access Swagger UI at \033[1mhttp://localhost:" + str(port) + "/ui\033[1m 🔗")
    api.run(port=port, debug=debug, server=deployment_server)


## TODO: put the code for the different calls of your application here! 

def get_predict_drug_disease(drug, disease):
    """Get associations predictions for drug-disease pairs
    
    :param drug: Drug of the predicted association 
    :param disease: Disease of the predicted association 
    :return: Prediction result object with score
    """
    # similarity_scores = get_drug_disease_similarities()
    prediction_result = {'drug' : drug, 'disease': disease, 'score': 0.8}
    return prediction_result or ('Not found', 404)