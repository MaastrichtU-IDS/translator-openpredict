import os
# import connexion
import logging
from datetime import datetime
from openpredict.openpredict_utils import init_openpredict_dir
from openpredict.rdf_utils import init_triplestore, retrieve_features, retrieve_models
from openpredict.openpredict_model import addEmbedding, get_predictions, get_similarities, load_similarity_embedding_models
from openpredict.reasonerapi_parser import typed_results_to_reasonerapi
# from openpredict.openapi import TRAPI_EXAMPLE, custom_openapi
from openpredict.openapi import TRAPI, TRAPI_EXAMPLE
# from flask_cors import CORS
# from flask_reverse_proxy_fix.middleware import ReverseProxyPrefixFix
import pkg_resources
# from gensim.models import KeyedVectors
# import asyncio
# import aiohttp
# from aiohttp import web

from fastapi import FastAPI, Body, Request, Response, Query
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from reasoner_pydantic import Query, Message
from typing import Optional, Dict

init_openpredict_dir()
init_triplestore()
# global all_emb_vectors
# all_emb_vectors = load_similarity_embedding_models()

# TODO: try using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

app = TRAPI(treats_model='openpredict-baseline-omim-drugbank')


@app.get("/meta_knowledge_graph", name="TRAPI query",
    description="Get the meta knowledge graph",
    # response_model=dict,
    tags=["trapi"],
)
def get_meta_knowledge_graph():
    """Get predicates and entities provided by the API

    :return: JSON with biolink entities
    """
    openpredict_predicates = {
        "edges": [
            {
                "object": "biolink:Disease",
                "predicate": "biolink:treats",
                "relations": [
                    "RO:0002434"
                ],
                "subject": "biolink:Drug"
            },
            {
                "object": "biolink:Drug",
                "predicate": "biolink:treated_by",
                "relations": [
                    "RO:0002434"
                ],
                "subject": "biolink:Disease"
            },
            {
                "object": "biolink:Entity",
                "predicate": "biolink:similar_to",
                # "relations": [
                #     "RO:0002434"
                # ],
                "subject": "biolink:Entity"
            },
        ],
        "nodes": {
            "biolink:Disease": {
                "id_prefixes": [
                    "OMIM"
                ]
            },
            "biolink:Drug": {
                "id_prefixes": [
                    "DRUGBANK"
                ]
            }
        }
    }

    return JSONResponse(openpredict_predicates)


@app.post("/query", name="TRAPI query",
    description="Get list of predicted associations for a given TRAPI query",
    response_model=dict,
    tags=["reasoner"],
)
def post_reasoner_predict(
        request_body: Query = Body(..., example=TRAPI_EXAMPLE)
    ):
    """Get predicted associations for a given ReasonerAPI query.

    :param request_body: The ReasonerStdAPI query in JSON
    :return: Predictions as a ReasonerStdAPI Message
    """
    # request_body = request.body()
    # query_graph = request_body["message"]["query_graph"]
    query_graph = request_body.message.query_graph.dict(exclude_none=True)
    print(query_graph)
    if len(query_graph["edges"]) == 0:
        return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
        # return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)
    if len(query_graph["edges"]) > 1:
        # Currently just return a empty result if multi-edges query
        return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
        # return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)

    # reasonerapi_response = typed_results_to_reasonerapi(request_body.dict(exclude_none=True))
    reasonerapi_response = typed_results_to_reasonerapi(
        request_body.dict(exclude_none=True), 
        app.treats_features,
        app.similarity_features,
        app.treats_classifier
    )

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

    return JSONResponse(reasonerapi_response) or ('Not found', 404)




# def post_embedding(apikey, types, emb_name, description, model_id):
#     """Post JSON embeddings via the API, with simple APIKEY authentication 
#     provided in environment variables 
#     """
#     # TODO: implement GitHub OAuth? https://github-flask.readthedocs.io/en/latest/
#     # Ignore the API key check if no env variable defined (for development)
#     print(os.getenv('OPENPREDICT_APIKEY'))
#     if os.getenv('OPENPREDICT_APIKEY') == apikey or os.getenv('OPENPREDICT_APIKEY') is None:
#         embedding_file = connexion.request.files['embedding_file']
#         print(emb_name, types)
#         run_id, scores = addEmbedding(
#             embedding_file, emb_name, types, description, model_id)
#         print('Embeddings uploaded')
#         # train_model(False)
#         return {
#             'status': 200,
#             'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(scores)
#         }
#     else:
#         return {'Forbidden': 403}


# def get_similarity(types='Both', drug_id=None, disease_id=None, model_id='openpredict-baseline-omim-drugbank', min_score=None, max_score=None, n_results=None):
#     """Get similar entites for a given entity CURIE.

#     :param entity: Search for predicted associations for this entity CURIE
#     :return: Prediction results object with score
#     """
#     time_start = datetime.now()

#     # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
#     concept_id = ''
#     if drug_id:
#         concept_id = drug_id
#     elif disease_id:
#         concept_id = disease_id
#     else:
#         return ('Bad request: provide a drugid or diseaseid', 400)

#     try:
#         emb_vectors = all_emb_vectors[model_id]
#         prediction_json, source_target_predictions = get_similarities(
#             types, concept_id, emb_vectors, min_score, max_score, n_results
#         )
#     except Exception as e:
#         print('Error processing ID ' + concept_id)
#         print(e)
#         return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

#     # try:
#     #     prediction_json = get_predictions(entity, model_id, score, n_results)
#     # except:
#     #     return "Not found", 404

#     # relation = "biolink:treated_by"
#     logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
#     return {'hits': prediction_json, 'count': len(prediction_json)}
#     # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)




# def get_predict(drug_id=None, disease_id=None, model_id='openpredict-baseline-omim-drugbank', min_score=None, max_score=None, n_results=None):
#     """Get predicted associations for a given entity CURIE.

#     :param entity: Search for predicted associations for this entity CURIE
#     :return: Prediction results object with score
#     """
#     time_start = datetime.now()

#     # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
#     concept_id = ''
#     if drug_id:
#         concept_id = drug_id
#     elif disease_id:
#         concept_id = disease_id
#     else:
#         return ('Bad request: provide a drugid or diseaseid', 400)

#     try:
#         prediction_json, source_target_predictions = get_predictions(
#             concept_id, model_id, min_score, max_score, n_results)
#     except Exception as e:
#         print('Error processing ID ' + concept_id)
#         print(e)
#         return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

#     # try:
#     #     prediction_json = get_predictions(entity, model_id, score, n_results)
#     # except:
#     #     return "Not found", 404

#     # relation = "biolink:treated_by"
#     logging.info('PredictRuntime: ' + str(datetime.now() - time_start))
#     return {'hits': prediction_json, 'count': len(prediction_json)}
#     # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)



# def get_predicates():
#     """DEPRECATED since 3.1.0, replaced by meta_knowledge_graph
#     Get predicates and entities provided by the API

#     :return: JSON with biolink entities
#     """
#     openpredict_predicates = {
#         "disease": {
#             "drug": [
#                 "treated_by"
#             ]
#         },
#         "drug": {
#             "disease": [
#                 "treats"
#             ]
#         }
#     }
#     return openpredict_predicates


# def get_features(type):
#     """Get features in the model

#     :return: JSON with features
#     """
#     return retrieve_features(type)


# def get_models():
#     """Get models with their scores and features

#     :return: JSON with models and features
#     """
#     return retrieve_models()


# async def async_reasoner_predict(request_body):
#     """Get predicted associations for a given ReasonerAPI query.

#     :param request_body: The ReasonerStdAPI query in JSON
#     :return: Predictions as a ReasonerStdAPI Message
#     """
#     return post_reasoner_predict(request_body)

# # TODO: get_predict wrapped in ReasonerStdApi
