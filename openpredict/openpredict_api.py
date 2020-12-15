import os
import connexion
import logging
from datetime import datetime
from openpredict.openpredict_utils import init_openpredict_dir
from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict.openpredict_model import addEmbedding, get_predictions
from openpredict.reasonerapi_parser import typed_results_to_reasonerapi
from flask_cors import CORS
from flask_reverse_proxy_fix.middleware import ReverseProxyPrefixFix

def start_api(port=8808, debug=False, start_spark=True):
    """Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

    :param port: Port of the OpenPredict API, defaults to 8808
    :param debug: Run in debug mode, defaults to False
    :param start_spark: Start a local Spark cluster, default to true
    """
    print("Starting the \033[1mTranslator OpenPredict API\033[0m 🔮🐍")

    init_openpredict_dir()

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
    
    
    api = connexion.App(__name__, options={"swagger_url": ""})
    # api = connexion.App(__name__, options={"swagger_url": ""}, arguments={'server_url': server_url})

    api.add_api('openapi.yml')
    # api.add_api('openapi.yml', arguments={'server_url': server_url}, validate_responses=True, options={"disable_servers_overwrite": True})

    # Add CORS support
    CORS(api.app)

    ## Fix to avoid empty list of servers for nginx-proxy deployments
    if os.getenv('LETSENCRYPT_HOST'):
        server_url='https://' + os.getenv('LETSENCRYPT_HOST')
        api.app.config['REVERSE_PROXY_PATH'] = server_url
        # api.app.config['REVERSE_PROXY_PATH'] = '/api'
        ReverseProxyPrefixFix(api.app)
    elif os.getenv('VIRTUAL_HOST'):
        server_url='http://' + os.getenv('VIRTUAL_HOST')
        api.app.config['REVERSE_PROXY_PATH'] = server_url
        # api.app.config['REVERSE_PROXY_PATH'] = '/api'
        ReverseProxyPrefixFix(api.app)

    # logging.info('Start spark:' + str(start_spark))
    # if start_spark:
    #     try:
    #         start_spark()
    #         logging.info('Started Spark locally')
    #     except:
    #         logging.info("Could not start Spark locally")

    print("Access Swagger UI at \033[1mhttp://localhost:" + str(port) + "\033[1m 🔗")
    api.run(host='0.0.0.0', port=port, debug=debug, server=deployment_server)


def post_embedding(types, emb_name, description, model_id):
    """Post JSON embeddings via the API, with simple APIKEY authentication 
    provided in environment variables 
    """
    # if os.getenv('OPENPREDICT_APIKEY') == apikey:
    print ('Post a new embeddings')
    if True:
        embedding_file = connexion.request.files['embedding_file']
        print (emb_name, types)
        run_id = addEmbedding(embedding_file, emb_name, types, description, model_id)
        print ('Embeddings uploaded')
        # train_model(False)
        return { 'Embeddings added for run ' + run_id: 200 }
    else:
        return { 'Forbidden': 403 }

def get_predict(model_id, drug_id=None, disease_id=None, min_score=None, max_score=None, n_results=None):
    """Get predicted associations for a given entity CURIE.
    
    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()

    # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions

    if drug_id:
        prediction_json, source_target_predictions = get_predictions(drug_id, model_id, min_score, max_score, n_results)
    elif disease_id:
        prediction_json, source_target_predictions = get_predictions(disease_id, model_id, min_score, max_score, n_results)
    else:
        return ('Bad request: provide a drugid or diseaseid', 400)

    # try:
    #     prediction_json = get_predictions(entity, model_id, score, n_results)
    # except:
    #     return "Not found", 404

    # relation = "biolink:treated_by"
    logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}
    # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)

def get_predicates():
    """Get predicates and entities provided by the API
    
    :return: JSON with biolink entities
    """
    openpredict_predicates = {
        "disease": {
            "drug": [
            "treated_by"
            ]
        }
    }
    return openpredict_predicates

def get_features(type):
    """Get features in the model
    
    :return: JSON with features
    """
    return retrieve_features(type)

def get_models():
    """Get models with their scores and features
    
    :return: JSON with models and features
    """
    return retrieve_models()

# TODO: get_predict wrapped in ReasonerStdApi
def post_reasoner_predict(request_body):
    """Get predicted associations for a given ReasonerAPI query.
    
    :param request_body: The ReasonerStdAPI query in JSON
    :return: Predictions as a ReasonerStdAPI Message
    """
    query_graph = request_body["message"]["query_graph"]
    model_id = 'openpredict-baseline-omim-drugbank'
    print(query_graph)
    if len(query_graph["edges"]) == 0:
        return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)
    if len(query_graph["edges"]) > 1:
        return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)

    reasonerapi_response = typed_results_to_reasonerapi(request_body, model_id)

    # TODO: populate edges/nodes with association predictions    
    #  Edge: {
    #     "id": "e50",
    #     "source_id": "MONDO:0021668",
    #     "target_id": "ChEMBL:CHEMBL560511",
    #     "type": "treated_by"
    #   }
    # Node: {
    #     "id": "ChEMBL:CHEMBL2106966",
    #     "name": "Piketoprofen",
    #     "type": "chemical_substance"
    #   },

    return reasonerapi_response or ('Not found', 404)