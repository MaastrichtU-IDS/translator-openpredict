import connexion
import logging
from datetime import datetime
from openpredict.predict_utils import get_predictions
from openpredict.train_utils import generate_feature_metadata, get_features_from_model
from openpredict.predict_model_omim_drugbank import addEmbedding, train_omim_drugbank_classifier
from openpredict.reasonerapi_parser import typed_results_to_reasonerapi
from rdflib import Graph, Literal, RDF, URIRef
import pkg_resources
from flask_cors import CORS

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

    # Add CORS support
    CORS(api.app)

    logging.info('Start spark:' + str(start_spark))
    if start_spark:
        try:
            start_spark()
            logging.info('Started Spark locally')
        except:
            logging.info("Could not start Spark locally")

    print("Access Swagger UI at \033[1mhttp://localhost:" + str(port) + "\033[1m 🔗")
    api.run(port=port, debug=debug, server=deployment_server)


def post_embedding(types, emb_name, description):
    embedding_file = connexion.request.files['embedding_file']
    print (emb_name, types)
    addEmbedding(embedding_file, emb_name, types, description)
    print ('Embeddings uploaded')
    # train_omim_drugbank_classifier(False)
    return { 'Embeddings added': 200 }
    # Code for the different calls of the app

def get_predict(entity, classifier="Predict OMIM-DrugBank", score=None, n_results=None):
    """Get predicted associations for a given entity CURIE.
    
    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()

    try:
        prediction_json = get_predictions(entity, classifier, score, n_results)
    except:
        return "Not found", 404

    relation = "biolink:treated_by"
    logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)

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
    return get_features_from_model(type)

def get_models():
    """Get models with their scores and features
    
    :return: JSON with models and features
    """
    g = Graph()
    g.parse(pkg_resources.resource_filename('openpredict', 'data/openpredict-metadata.ttl'), format="ttl")

    sparql_get_scores = """PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX mls: <http://www.w3.org/ns/mls#>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX openpredict: <https://w3id.org/openpredict/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xml: <http://www.w3.org/XML/1998/namespace>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT DISTINCT ?model ?label ?generatedAtTime ?features ?accuracy ?average_precision ?f1 ?precision ?recall ?roc_auc
        WHERE {
            ?model a mls:ModelEvaluation ;
                rdfs:label ?label ;
                prov:generatedAtTime ?generatedAtTime ;
                openpredict:has_features ?features .
            ?model mls:specifiedBy [a mls:EvaluationMeasure ; 
                        rdfs:label "accuracy" ;
                        mls:hasValue ?accuracy ] .
            ?model mls:specifiedBy [ a mls:EvaluationMeasure ; 
                    rdfs:label "precision" ;
                    mls:hasValue ?precision ] .
            ?model mls:specifiedBy [ a mls:EvaluationMeasure ; 
                    rdfs:label "f1" ;
                    mls:hasValue ?f1 ] .
            ?model mls:specifiedBy [ a mls:EvaluationMeasure ; 
                    rdfs:label "recall" ;
                    mls:hasValue ?recall ] .
            ?model mls:specifiedBy [ a mls:EvaluationMeasure ; 
                    rdfs:label "roc_auc" ;
                    mls:hasValue ?roc_auc ] .
            ?model mls:specifiedBy [ a mls:EvaluationMeasure ; 
                    rdfs:label "average_precision" ;
                    mls:hasValue ?average_precision ] .
        }
        """
    qres = g.query(sparql_get_scores)
    features_json = {}
    for row in qres:
        print(row.label)
        if row.model in features_json:
            features_json[row.model]['features'].append(row.features)
        else:
            features_json[row.model] = {
                "label": row.label,
                "generatedAtTime": row.generatedAtTime,
                'features': [row.features],
                'accuracy': row.accuracy,
                'average_precision': row.average_precision,
                'f1': row.f1,
                'precision': row.precision,
                'recall': row.recall,
                'roc_auc': row.roc_auc
            }
    return features_json

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