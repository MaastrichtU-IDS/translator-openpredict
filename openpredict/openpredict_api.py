import connexion
import logging
from datetime import datetime
from openpredict.predict_utils import get_predictions
from openpredict.reasonerapi_parser import typed_results_to_reasonerapi

# import openpredict.utils

def start_spark():
    """Start local Spark cluster when possible to improve performance
    """
    logging.info("Trying to find a Spark cluster...")
    import findspark
    from pyspark import SparkConf, SparkContext
    findspark.init()

    config = SparkConf()
    config.setMaster("local[*]")
    config.set("spark.executor.memory", "5g")
    config.set('spark.driver.memory', '5g')
    config.set("spark.memory.offHeap.enabled",True)
    config.set("spark.memory.offHeap.size","5g") 
    sc = SparkContext(conf=config, appName="OpenPredict")
    print (sc)

def start_api(port=8808, server_url='/', debug=False, start_spark=True):
    """Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

    :param port: Port of the OpenPredict API, defaults to 8808
    :param debug: Run in debug mode, defaults to False
    :param start_spark: Start a local Spark cluster, default to true
    """
    print("Starting the \033[1mTranslator OpenPredict API\033[0m 🔮🐍")

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

    api.add_api('openapi.yml', arguments={'server_url': server_url})
    # api.add_api('openapi.yml', arguments={'server_url': server_url}, validate_responses=True)

    logging.info('Start spark:' + str(start_spark))
    if start_spark:
        try:
            start_spark()
            logging.info('Started Spark locally')
        except:
            logging.info("Could not start Spark locally")

    print("Access Swagger UI at \033[1mhttp://localhost:" + str(port) + "\033[1m 🔗")
    api.run(port=port, debug=debug, server=deployment_server)



### Code for the different calls of the app

def get_predict(entity, classifier="OpenPredict OMIM-DrugBank", score=None, limit=None):
    """Get predicted associations for a given entity CURIE.
    
    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()
    # classifier: OpenPredict OMIM-DrugBank
    print("Using classifier: " + classifier)
    # prediction_json = get_predictions(entity, classifier, score, limit)
    try:
        prediction_json = get_predictions(entity, classifier, score, limit)
    except:
        return "Not found", 404

    relation = "biolink:treated_by"
    logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)

def predicates_get():
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

# TODO: get_predict wrapped in ReasonerStdApi
def post_reasoner_predict(request_body):
    """Get predicted associations for a given ReasonerAPI query.
    
    :param request_body: The ReasonerStdAPI query in JSON
    :return: Predictions as a ReasonerStdAPI Message
    """
    query_graph = request_body["message"]["query_graph"]
    print(query_graph)
    if len(query_graph["edges"]) == 0:
        return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)
    if len(query_graph["edges"]) > 1:
        return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)

    reasonerapi_response = typed_results_to_reasonerapi(request_body)

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