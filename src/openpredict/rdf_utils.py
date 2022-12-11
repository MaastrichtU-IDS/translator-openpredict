import os
import uuid
from datetime import datetime

from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DC, RDFS, XSD
from SPARQLWrapper import JSON, POST, SPARQLWrapper

from openpredict.config import settings

if not settings.OPENPREDICT_DATA_DIR.endswith('/'):
    settings.OPENPREDICT_DATA_DIR += '/'
RDF_DATA_PATH = settings.OPENPREDICT_DATA_DIR + 'openpredict-metadata.ttl'


OPENPREDICT_GRAPH = 'https://w3id.org/openpredict/graph'
OPENPREDICT_NAMESPACE = 'https://w3id.org/openpredict/'
BIOLINK = Namespace("https://w3id.org/biolink/vocab/")

RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SCHEMA = Namespace("http://schema.org/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
PROV = Namespace("http://www.w3.org/ns/prov#")
MLS = Namespace("http://www.w3.org/ns/mls#")
OPENPREDICT = Namespace("https://w3id.org/openpredict/")

# Get SPARQL endpoint credentials from environment variables
SPARQL_ENDPOINT_PASSWORD = os.getenv('SPARQL_PASSWORD')
SPARQL_ENDPOINT_USERNAME = os.getenv('SPARQL_USERNAME')
SPARQL_ENDPOINT_URL = os.getenv('SPARQL_ENDPOINT_URL')
SPARQL_ENDPOINT_UPDATE_URL = os.getenv('SPARQL_ENDPOINT_UPDATE_URL')

# Default credentials for dev (if no environment variables provided)
if not SPARQL_ENDPOINT_USERNAME:
    # SPARQL_ENDPOINT_USERNAME='import_user'
    SPARQL_ENDPOINT_USERNAME = 'dba'
if not SPARQL_ENDPOINT_PASSWORD:
    SPARQL_ENDPOINT_PASSWORD = 'dba'
# if not SPARQL_ENDPOINT_URL:
#    SPARQL_ENDPOINT_URL='http://localhost:8890/sparql'
# SPARQL_ENDPOINT_URL='https://graphdb.dumontierlab.com/repositories/translator-openpredict-dev'
# if not SPARQL_ENDPOINT_UPDATE_URL:
#    SPARQL_ENDPOINT_UPDATE_URL = 'http://localhost:8890/sparql'
    # SPARQL_ENDPOINT_UPDATE_URL='https://graphdb.dumontierlab.com/repositories/translator-openpredict-dev/statements'

# Uncomment this line to test OpenPredict in dev mode using a RDF file instead of a SPARQL endpoint
SPARQL_ENDPOINT_URL=None


def insert_graph_in_sparql_endpoint(g):
    """Insert rdflib graph in a Update SPARQL endpoint using SPARQLWrapper

    :param g: rdflib graph to insert
    :return: SPARQL update query result
    """
    if SPARQL_ENDPOINT_URL:
        sparql = SPARQLWrapper(SPARQL_ENDPOINT_UPDATE_URL)
        sparql.setMethod(POST)
        # sparql.setHTTPAuth(BASIC)
        sparql.setCredentials(SPARQL_ENDPOINT_USERNAME,
                              SPARQL_ENDPOINT_PASSWORD)
        query = """INSERT DATA {{ GRAPH  <{graph}>
        {{
        {ntriples}
        }}
        }}
        """.format(ntriples=g.serialize(format='nt').decode('utf-8'), graph=OPENPREDICT_GRAPH)

        sparql.setQuery(query)
        return sparql.query()
    else:
        # If no SPARQL endpoint provided we store to the RDF file in data/openpredict-metadata.ttl (working)
        graph_from_file = Graph()
        graph_from_file.parse(RDF_DATA_PATH, format="ttl")
        # graph_from_file.parse(g.serialize(format='turtle').decode('utf-8'), format="ttl")
        graph_from_file = graph_from_file + g
        graph_from_file.serialize(RDF_DATA_PATH, format='turtle')


def query_sparql_endpoint(query, parameters=[]):
    """Run select SPARQL query against SPARQL endpoint

    :param query: SPARQL query as a string
    :return: Object containing the result bindings
    """
    if SPARQL_ENDPOINT_URL:
        sparql = SPARQLWrapper(SPARQL_ENDPOINT_URL)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        results = sparql.query().convert()
        # print('SPARQLWrapper Results:')
        # print(results["results"]["bindings"])
        return results["results"]["bindings"]
    else:
        # Trying to SPARQL query a RDF file directly, to avoid using triplestores in dev (not working)
        # Docs: https://rdflib.readthedocs.io/en/stable/intro_to_sparql.html
        # Examples: https://github.com/RDFLib/rdflib/tree/master/examples
        # Use SPARQLStore? https://github.com/RDFLib/rdflib/blob/master/examples/sparqlstore_example.py
        # But this would require to rewrite all SPARQL query resolution to use rdflib response object
        # Which miss the informations about which SPARQL variables (just returns rows of results without variable bind)
        g = Graph()
        g.parse(RDF_DATA_PATH, format="ttl")
        # print('RDF data len')
        # print(len(g))
        # print(query)
        qres = g.query(query)
        # print('query done')
        # print(qres)
        results = []
        for row in qres:
            # TODO: row.asdict()
            result = {}
            for i, p in enumerate(parameters):
                result[p] = {}
                result[p]['value'] = str(row[p])
            results.append(result)
            # How can we iterate over the variable defined in the SPARQL query?
            # It only returns the results, without the variables list
            # Does not seems possible: https://dokk.org/documentation/rdflib/3.2.0/gettingstarted/#run-a-query
            # print(row.run)
            # or row["s"]
            # or row[rdflib.Variable("s")]
            # TODO: create an object similar to SPARQLWrapper
            # result[variable]['value']
        # print(results)
        return results


# def init_triplestore():
#     """Only initialized the triplestore if no run for openpredict-baseline-omim-drugbank can be found.
#     Init using the data/openpredict-metadata.ttl RDF file
#     """
#     # check_baseline_run_query = """SELECT DISTINCT ?runType
#     # WHERE {
#     #     <https://w3id.org/openpredict/run/openpredict-baseline-omim-drugbank> a ?runType
#     # } LIMIT 10
#     # """
#     # results = query_sparql_endpoint(check_baseline_run_query, parameters=['runType'])
#     # if (len(results) < 1):
#     g = Graph()
#     g.parse('openpredict/data/openpredict-metadata.ttl', format="ttl")
#     insert_graph_in_sparql_endpoint(g)
#     print('Triplestore initialized at ' + SPARQL_ENDPOINT_UPDATE_URL)


def add_feature_metadata(id, description, type):
    """Generate RDF metadata for a feature

    :param id: if used to identify the feature
    :param description: feature description
    :param type: feature type
    :return: rdflib graph after loading the feature
    """
    g = Graph()

    feature_uri = URIRef(OPENPREDICT_NAMESPACE + 'feature/' + id)
    g.add((feature_uri, RDF.type, MLS['Feature']))
    g.add((feature_uri, DC.identifier, Literal(id)))
    g.add((feature_uri, DC.description, Literal(description)))
    g.add((feature_uri, OPENPREDICT['embedding_type'], Literal(type)))

    insert_graph_in_sparql_endpoint(g)
    return str(feature_uri)


def get_run_id(run_id=None):
    if not run_id:
        # Generate random UUID for the run ID
        run_id = str(uuid.uuid1())
    return run_id


def add_run_metadata(scores, model_features, hyper_params, run_id=None):
    """Generate RDF metadata for a classifier and save it in data/openpredict-metadata.ttl, based on OpenPredict model:
    https://github.com/fair-workflows/openpredict/blob/master/data/rdf/results_disjoint_lr.nq

    :param scores: scores
    :param model_features: List of features in the model
    :param label: label of the classifier
    :return: Run id
    """
    g = Graph()
    if not run_id:
        # Generate random UUID for the run ID
        run_id = str(uuid.uuid1())

    run_uri = URIRef(OPENPREDICT_NAMESPACE + 'run/' + run_id)
    run_prop_prefix = OPENPREDICT_NAMESPACE + run_id + "/"
    evaluation_uri = URIRef(OPENPREDICT_NAMESPACE +
                            'run/' + run_id + '/ModelEvaluation')
    # The same for all run:
    implementation_uri = URIRef(
        OPENPREDICT_NAMESPACE + 'implementation/OpenPredict')

    # Add Run metadata
    g.add((run_uri, RDF.type, MLS['Run']))
    g.add((run_uri, DC.identifier, Literal(run_id)))
    g.add((run_uri, PROV['generatedAtTime'], Literal(
        datetime.now(), datatype=XSD.dateTime)))
    g.add((run_uri, MLS['realizes'], OPENPREDICT['LogisticRegression']))
    g.add((run_uri, MLS['executes'], implementation_uri))
    g.add((run_uri, MLS['hasOutput'], evaluation_uri))
    g.add((run_uri, MLS['hasOutput'], URIRef(run_prop_prefix + 'Model')))

    # Add Model, should we point it to the generated model?
    g.add((URIRef(run_prop_prefix + 'Model'), RDF.type, MLS['Model']))

    # Add implementation metadata
    g.add((OPENPREDICT['LogisticRegression'], RDF.type, MLS['Algorithm']))
    g.add((implementation_uri, RDF.type, MLS['Implementation']))
    g.add((implementation_uri, MLS['implements'],
          OPENPREDICT['LogisticRegression']))

    # Add HyperParameters and their settings to implementation
    for hyperparam, hyperparam_setting in hyper_params.items():
        hyperparam_uri = URIRef(
            OPENPREDICT_NAMESPACE + 'HyperParameter/' + hyperparam)
        g.add((implementation_uri, MLS['hasHyperParameter'], hyperparam_uri))
        g.add((hyperparam_uri, RDF.type, MLS['HyperParameter']))
        g.add((hyperparam_uri, RDFS.label, Literal(hyperparam)))

        hyperparam_setting_uri = URIRef(
            OPENPREDICT_NAMESPACE + 'HyperParameterSetting/' + hyperparam)
        g.add(
            (implementation_uri, MLS['hasHyperParameter'], hyperparam_setting_uri))
        g.add((hyperparam_setting_uri, RDF.type, MLS['HyperParameterSetting']))
        g.add((hyperparam_setting_uri, MLS['specifiedBy'], hyperparam_uri))
        g.add((hyperparam_setting_uri,
              MLS['hasValue'], Literal(hyperparam_setting)))
        g.add((run_uri, MLS['hasInput'], hyperparam_setting_uri))

    # TODO: improve how we retrieve features
    for feature in model_features:
        feature_uri = URIRef(feature)
        # feature_uri = URIRef(OPENPREDICT_NAMESPACE + 'feature/' + feature)
        # g.add((run_uri, OPENPREDICT['has_features'], feature_uri))
        g.add((run_uri, MLS['hasInput'], feature_uri))

    # TODO: those 2 triples are for the PLEX ontology
    g.add((evaluation_uri, RDF.type, PROV['Entity']))
    g.add((evaluation_uri, PROV['wasGeneratedBy'], run_uri))

    # Add scores as EvaluationMeasures
    g.add((evaluation_uri, RDF.type, MLS['ModelEvaluation']))
    for key in scores.keys():
        key_uri = URIRef(run_prop_prefix + key)
        g.add((evaluation_uri, MLS['specifiedBy'], key_uri))
        g.add((key_uri, RDF.type, MLS['EvaluationMeasure']))
        g.add((key_uri, RDFS.label, Literal(key)))
        g.add((key_uri, MLS['hasValue'], Literal(
            scores[key], datatype=XSD.double)))
        # TODO: The Example 1 puts hasValue directly in the ModelEvaluation
        # but that prevents to provide multiple values for 1 evaluation
        # http://ml-schema.github.io/documentation/ML%20Schema.html#overview

    insert_graph_in_sparql_endpoint(g)
    return run_id


def retrieve_features(type='Both', run_id=None):
    """Get features in the ML model

    :param type: type of the feature (Both, Drug, Disease)
    :return: JSON with features
    """
    if run_id:
        sparql_feature_for_run = """PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX mls: <http://www.w3.org/ns/mls#>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX openpredict: <https://w3id.org/openpredict/>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT DISTINCT ?feature ?featureId ?featureDescription ?embeddingType
            WHERE {
                ?run a mls:Run ;
                    dc:identifier \"""" + run_id + """\" ;
                    mls:hasInput ?feature .
                ?feature dc:identifier ?featureId ;
                    <https://w3id.org/openpredict/embedding_type> ?embeddingType ;
                    dc:description ?featureDescription .
            }"""
        results = query_sparql_endpoint(sparql_feature_for_run, parameters=[
                                        'feature', 'featureId', 'featureDescription', 'embeddingType'])
        # print(results)

        features_json = {}
        for result in results:
            features_json[result['feature']['value']] = {
                "id": result['featureId']['value'],
                "description": result['featureDescription']['value'],
                "type": result['embeddingType']['value']
            }

    else:
        type_filter = ''
        if (type != "Both"):
            type_filter = 'FILTER(?embeddingType = "' + type + '")'

        query = """SELECT DISTINCT ?id ?description ?embeddingType ?feature
            WHERE {{
                ?feature a <http://www.w3.org/ns/mls#Feature> ;
                    <http://purl.org/dc/elements/1.1/identifier> ?id ;
                    <https://w3id.org/openpredict/embedding_type> ?embeddingType ;
                    <http://purl.org/dc/elements/1.1/description> ?description .
                {type_filter}
            }}
            """.format(type_filter=type_filter)

        results = query_sparql_endpoint(
            query, parameters=['id', 'description', 'embeddingType', 'feature'])
        # print(results)

        features_json = {}
        for result in results:
            features_json[result['feature']['value']] = {
                "id": result['id']['value'],
                "description": result['description']['value'],
                "type": result['embeddingType']['value']
            }
    return features_json


def retrieve_models():
    """Get models with their scores and features

    :return: JSON with models and features
    """
    sparql_get_scores = """PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX mls: <http://www.w3.org/ns/mls#>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX openpredict: <https://w3id.org/openpredict/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT DISTINCT ?run ?runId ?generatedAtTime ?featureId ?accuracy ?average_precision ?f1 ?precision ?recall ?roc_auc
        WHERE {
    		?run a mls:Run ;
                dc:identifier ?runId ;
           		prov:generatedAtTime ?generatedAtTime ;
                mls:hasInput ?features ;
            	mls:hasOutput ?evaluation .
            ?evaluation a mls:ModelEvaluation  .
            ?features dc:identifier ?featureId .

            ?evaluation mls:specifiedBy [a mls:EvaluationMeasure ;
                        rdfs:label "accuracy" ;
                        mls:hasValue ?accuracy ] .
            ?evaluation mls:specifiedBy [ a mls:EvaluationMeasure ;
                    rdfs:label "precision" ;
                    mls:hasValue ?precision ] .
            ?evaluation mls:specifiedBy [ a mls:EvaluationMeasure ;
                    rdfs:label "f1" ;
                    mls:hasValue ?f1 ] .
            ?evaluation mls:specifiedBy [ a mls:EvaluationMeasure ;
                    rdfs:label "recall" ;
                    mls:hasValue ?recall ] .
            ?evaluation mls:specifiedBy [ a mls:EvaluationMeasure ;
                    rdfs:label "roc_auc" ;
                    mls:hasValue ?roc_auc ] .
            ?evaluation mls:specifiedBy [ a mls:EvaluationMeasure ;
                    rdfs:label "average_precision" ;
                    mls:hasValue ?average_precision ] .
        }
        """

    results = query_sparql_endpoint(sparql_get_scores,
                                    parameters=['run', 'runId', 'generatedAtTime', 'featureId', 'accuracy',
                                                'average_precision', 'f1', 'precision', 'recall', 'roc_auc'])
    models_json = {}
    for result in results:
        if result['run']['value'] in models_json:
            models_json[result['run']['value']]['features'].append(
                result['featureId']['value'])
        else:
            models_json[result['run']['value']] = {
                "id": result['runId']['value'],
                "generatedAtTime": result['generatedAtTime']['value'],
                'features': [result['featureId']['value']],
                'accuracy': result['accuracy']['value'],
                'average_precision': result['average_precision']['value'],
                'f1': result['f1']['value'],
                'precision': result['precision']['value'],
                'recall': result['recall']['value'],
                'roc_auc': result['roc_auc']['value']
            }

        # We could create an object with feature description instead of passing just the ID
        # features_json[result['id']['value']] = {
        #     "description": result['description']['value'],
        #     "type": result['embeddingType']['value']
        # }
    return models_json
