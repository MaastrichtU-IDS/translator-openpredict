import glob
import os
import pickle
import uuid

import pandas as pd
import requests
from gensim.models import KeyedVectors
from rdflib import Graph
from SPARQLWrapper import JSON, SPARQLWrapper
from trapi_predict_kit import log, normalize_id_to_translator, settings


def is_accepted_id(id_to_check):
    return id_to_check.lower().startswith("omim") or id_to_check.lower().startswith("drugbank")


def resolve_ids_with_nodenormalization_api(resolve_ids_list):
    trapi_to_supported = {}
    supported_to_trapi = {}
    ids_to_normalize = []
    for id_to_resolve in resolve_ids_list:
        if is_accepted_id(id_to_resolve):
            supported_to_trapi[id_to_resolve] = id_to_resolve
            trapi_to_supported[id_to_resolve] = id_to_resolve
        else:
            ids_to_normalize.append(id_to_resolve)

    # Query Translator NodeNormalization API to convert IDs to OMIM/DrugBank IDs
    if len(ids_to_normalize) > 0:
        try:
            resolve_curies = requests.get(
                "https://nodenormalization-sri.renci.org/get_normalized_nodes",
                params={"curie": ids_to_normalize},
                timeout=settings.TIMEOUT,
            )
            # Get corresponding OMIM IDs for MONDO IDs if match
            resp = resolve_curies.json()
            for resolved_id, alt_ids in resp.items():
                for alt_id in alt_ids["equivalent_identifiers"]:
                    if is_accepted_id(str(alt_id["identifier"])):
                        main_id = str(alt_id["identifier"])
                        # NOTE: fix issue when NodeNorm returns OMIM.PS: instead of OMIM:
                        if main_id.lower().startswith("omim"):
                            main_id = "OMIM:" + main_id.split(":", 1)[1]
                        trapi_to_supported[resolved_id] = main_id
                        supported_to_trapi[main_id] = resolved_id
        except Exception:
            log.warn("Error querying the NodeNormalization API, using the original IDs")
    # log.info(f"Resolved: {resolve_ids_list} to {resolved_ids_object}")
    return trapi_to_supported, supported_to_trapi


def resolve_id(id_to_resolve, resolved_ids_object):
    if id_to_resolve in resolved_ids_object:
        return resolved_ids_object[id_to_resolve]
    return id_to_resolve



def get_openpredict_dir(subfolder: str = "") -> str:
    """Return the full path to the provided files in the OpenPredict data folder
    Where models and features for runs are stored
    """
    data_dir = os.getenv("OPENPREDICT_DATA_DIR", os.path.join(os.getcwd(), "data"))
    if not data_dir.endswith("/"):
        data_dir += "/"
    return data_dir + subfolder


default_model_id = "openpredict_baseline"
embedding_folder = os.path.join(get_openpredict_dir(), 'embedding')


def load_features_embeddings(model_id: str = default_model_id):
    """Load embeddings model for treats and treated_by"""
    print(f"📥 Loading treatment features for model {str(model_id)}")
    return pickle.load(open(f"{get_openpredict_dir()}features/{model_id}_features.pickle", "rb"))


def load_similarity_embeddings(model_id: str = default_model_id):
    """Load embeddings model for similarity"""
    feature_path = os.path.join(embedding_folder, model_id)
    log.info(f"📥 Loading similarity features from {feature_path}")
    emb_vectors = KeyedVectors.load_word2vec_format(feature_path)
    return emb_vectors


# TODO: not used
MISSING_IDS = set()
def convert_baseline_features_ids():
    """Convert IDs to use Translator preferred IDs when building the baseline model from scratch"""
    baseline_features_folder = os.path.join(get_openpredict_dir(), "baseline_features")
    drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv',
                    'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    drugfeatfiles = [
        os.path.join(baseline_features_folder, fn) for fn in drugfeatfiles
    ]
    diseasefeatfiles = [
        os.path.join(baseline_features_folder, fn) for fn in diseasefeatfiles
    ]

    # Prepare drug-disease dictionary
    drugDiseaseKnown = pd.read_csv(os.path.join(get_openpredict_dir(), 'resources', 'openpredict-omim-drug.csv'),delimiter=',')
    drugDiseaseKnown.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)

    drugs_set = set()
    diseases_set = set()
    drugs_set.update(drugDiseaseKnown['Drug'].tolist())
    diseases_set.update(drugDiseaseKnown['Disease'].tolist())

    for csv_file in drugfeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        drugs_set.update(df['Drug1'].tolist())
        drugs_set.update(df['Drug2'].tolist())

    for csv_file in diseasefeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        diseases_set.update(df['Disease1'].tolist())
        diseases_set.update(df['Disease2'].tolist())

    diseases_set = ['OMIM:{}'.format(disease) for disease in diseases_set]
    drugs_set = ['DRUGBANK:{}'.format(drug) for drug in drugs_set]

    diseases_mappings = normalize_id_to_translator(diseases_set)
    drugs_mappings = normalize_id_to_translator(drugs_set)

    log.info('Finished API queries')
    # Replace Ids with translator IDs in kown drug disease associations
    drugDiseaseKnown["Drug"] = drugDiseaseKnown["Drug"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row)     )
    drugDiseaseKnown["Disease"] = drugDiseaseKnown["Disease"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
    drugDiseaseKnown.to_csv('openpredict/data/resources/known-drug-diseases.csv', index=False)

    # Replace IDs in drugs baseline features files
    for csv_file in drugfeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        df["Drug1"] = df["Drug1"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row) )
        df["Drug2"] = df["Drug2"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row) )
        df.to_csv(csv_file.replace('/baseline_features/', '/translator_features/'), index=False)

    # Replace IDs in diseases baseline features files
    for csv_file in diseasefeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        df["Disease1"] = df["Disease1"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
        df["Disease2"] = df["Disease2"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
        df.to_csv(csv_file.replace('/baseline_features/', '/translator_features/'), index=False)

    log.warn(f"❌️ Missing IDs: {', '.join(MISSING_IDS)}")

    # drugs_set.add(2)
    # drugs_set.update([2, 3, 4])
    # Extract the dataframes col1 and 2 to a unique list
    # Add those list to the drugs and diseases sets
    # Convert the set/list it using normalize_id_to_translator(ids_list)
    # Update all dataframes using the created mappings
    # And store to baseline_translator


def map_id_to_translator(mapping_obj, source_id):
    try:
        return mapping_obj[source_id]
    except:
        MISSING_IDS.add(source_id)
        return source_id



# Get SPARQL endpoint credentials from environment variables
SPARQL_ENDPOINT_PASSWORD = os.getenv("SPARQL_PASSWORD")
SPARQL_ENDPOINT_USERNAME = os.getenv("SPARQL_USERNAME")
SPARQL_ENDPOINT_URL = os.getenv("SPARQL_ENDPOINT_URL")
SPARQL_ENDPOINT_UPDATE_URL = os.getenv("SPARQL_ENDPOINT_UPDATE_URL")

# Default credentials for dev (if no environment variables provided)
if not SPARQL_ENDPOINT_USERNAME:
    # SPARQL_ENDPOINT_USERNAME='import_user'
    SPARQL_ENDPOINT_USERNAME = "dba"
if not SPARQL_ENDPOINT_PASSWORD:
    SPARQL_ENDPOINT_PASSWORD = "dba"
# if not SPARQL_ENDPOINT_URL:
#    SPARQL_ENDPOINT_URL='http://localhost:8890/sparql'
# SPARQL_ENDPOINT_URL='https://graphdb.dumontierlab.com/repositories/translator-openpredict-dev'
# if not SPARQL_ENDPOINT_UPDATE_URL:
#    SPARQL_ENDPOINT_UPDATE_URL = 'http://localhost:8890/sparql'
# SPARQL_ENDPOINT_UPDATE_URL='https://graphdb.dumontierlab.com/repositories/translator-openpredict-dev/statements'

# Uncomment this line to test OpenPredict in dev mode using a RDF file instead of a SPARQL endpoint
SPARQL_ENDPOINT_URL = None


def get_models_graph(models_dir: str = "models"):
    """Helper function to get a graph with the RDF from all models given in a list"""
    g = Graph()

    for file in glob.glob(f"{models_dir}/*.ttl"):
        g.parse(file)

    # for loaded_model in models_list:
    #     g.parse(f"{loaded_model['model']}.ttl")
    #     # g.parse(f"{os.getcwd()}/{loaded_model['model']}.ttl")
    return g


def query_sparql_endpoint(query, g, parameters=[]):
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
        qres = g.query(query)
        results = []
        for row in qres:
            result = {}
            for _i, p in enumerate(parameters):
                result[p] = {}
                result[p]["value"] = str(row[p])
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
#     """Only initialized the triplestore if no run for openpredict_baseline can be found.
#     Init using the data/openpredict-metadata.ttl RDF file
#     """
#     # check_baseline_run_query = """SELECT DISTINCT ?runType
#     # WHERE {
#     #     <https://w3id.org/openpredict/run/openpredict_baseline> a ?runType
#     # } LIMIT 10
#     # """
#     # results = query_sparql_endpoint(check_baseline_run_query, parameters=['runType'])
#     # if (len(results) < 1):
#     g = Graph()
#     g.parse('openpredict/data/openpredict-metadata.ttl', format="ttl")
#     insert_graph_in_sparql_endpoint(g)
#     print('Triplestore initialized at ' + SPARQL_ENDPOINT_UPDATE_URL)


def get_run_id(run_id=None):
    if not run_id:
        # Generate random UUID for the run ID
        run_id = str(uuid.uuid1())
    return run_id



def retrieve_features(g, type="Both", run_id=None):
    """Get features in the ML model

    :param type: type of the feature (Both, Drug, Disease)
    :return: JSON with features
    """
    if run_id:
        sparql_feature_for_run = (
            """PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX mls: <http://www.w3.org/ns/mls#>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX openpredict: <https://w3id.org/openpredict/>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT DISTINCT ?feature ?featureId
            WHERE {
                ?run a mls:Run ;
                    dc:identifier \""""
            + run_id
            + """\" ;
                    mls:hasInput ?feature .
                ?feature dc:identifier ?featureId .
            }"""
        )
        # <https://w3id.org/openpredict/embedding_type> ?embeddingType ;
        #         dc:description ?featureDescription .
        results = query_sparql_endpoint(sparql_feature_for_run, g, parameters=["feature", "featureId"])
        # print(results)

        features_json = {}
        for result in results:
            features_json[result["feature"]["value"]] = {
                "id": result["featureId"]["value"],
            }

    else:
        # type_filter = ''
        # if (type != "Both"):
        #     type_filter = 'FILTER(?embeddingType = "' + type + '")'

        query = """SELECT DISTINCT ?id ?feature
            WHERE {{
                ?feature a <http://www.w3.org/ns/mls#Feature> ;
                    <http://purl.org/dc/elements/1.1/identifier> ?id .
            }}
            """
        # {type_filter} .format(type_filter=type_filter)
        results = query_sparql_endpoint(query, g, parameters=["id", "feature"])
        # print(results)

        features_json = {}
        for result in results:
            features_json[result["feature"]["value"]] = {
                "id": result["id"]["value"],
            }
    return features_json


def retrieve_models(g):
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

    results = query_sparql_endpoint(
        sparql_get_scores,
        g,
        parameters=[
            "run",
            "runId",
            "generatedAtTime",
            "featureId",
            "accuracy",
            "average_precision",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ],
    )
    models_json = {}
    for result in results:
        if result["run"]["value"] in models_json:
            models_json[result["run"]["value"]]["features"].append(result["featureId"]["value"])
        else:
            models_json[result["run"]["value"]] = {
                "id": result["runId"]["value"],
                "generatedAtTime": result["generatedAtTime"]["value"],
                "features": [result["featureId"]["value"]],
                "accuracy": result["accuracy"]["value"],
                "average_precision": result["average_precision"]["value"],
                "f1": result["f1"]["value"],
                "precision": result["precision"]["value"],
                "recall": result["recall"]["value"],
                "roc_auc": result["roc_auc"]["value"],
            }

        # We could create an object with feature description instead of passing just the ID
        # features_json[result['id']['value']] = {
        #     "description": result['description']['value'],
        #     "type": result['embeddingType']['value']
        # }
    return models_json
