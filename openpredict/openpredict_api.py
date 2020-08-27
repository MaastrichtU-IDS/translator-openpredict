import connexion
import json
import logging
from datetime import datetime
from openpredict.openpredict_omim_drugbank import query_omim_drugbank_classifier
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

def start_api(port=8808, debug=False, start_spark=True):
    """Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

    :param port: Port of the OpenPredict API, defaults to 8808
    :param debug: Run in debug mode, defaults to False
    :param start_spark: Start a local Spark cluster, default to true
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

def get_predict(entity, classifier="OpenPredict OMIM-DrugBank"):
    """Get predicted associations for a given entity CURIE.
    
    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()
    # classifier: OpenPredict OMIM-DrugBank
    print("Using classifier: " + classifier)
    prediction_json=json.loads(query_omim_drugbank_classifier(entity))
    
    # Expected? prediction_json = {
    #    'results': [{'source' : entity, 'target': 'associated drug 1', 'score': 0.8}],
    #    'count': 1
    #}
    relation = "biolink:treated_by"
    logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)


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