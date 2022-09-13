import os
from datetime import datetime
from typing import Dict, Optional,Union

from fastapi import Body, FastAPI, File, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import Field
from reasoner_pydantic import Message
from reasoner_pydantic import Query as TrapiQuery

from openpredict.evidence_path import do_evidence_path
from openpredict.openapi import TRAPI, TRAPI_EXAMPLE, EmbeddingTypes, FeatureTypesDiseases, FeatureTypesDrugs, SimilarityTypes
from openpredict.openpredict_model import (
    addEmbedding,
    get_predictions,
    get_similarities,
    load_similarity_embeddings,
)
from openpredict.rdf_utils import init_triplestore, retrieve_features, retrieve_models
from openpredict.trapi_parser import resolve_trapi_query
from openpredict.utils import init_openpredict_dir

# from gensim.models import KeyedVectors
# import asyncio
# import aiohttp
# from aiohttp import web
# import logging


init_openpredict_dir()
# init_triplestore()

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

# debug = os.getenv('DEV_MODE', False)

app = TRAPI(
    baseline_model_treatment='openpredict-baseline-omim-drugbank',
    baseline_model_similarity='drugs_fp_embed.txt',
)


@app.post("/query", name="TRAPI query",
    description="""The default example TRAPI query will give you a list of predicted potential drug treatments for a given disease

You can also try this query to retrieve similar entities to a given drug:

```json
{
    "message": {
        "query_graph": {
            "edges": {
                "e01": {
                    "object": "n1",
                    "predicates": [ "biolink:similar_to" ],
                    "subject": "n0"
                }
            },
            "nodes": {
                "n0": {
                    "categories": [ "biolink:Drug" ],
                    "ids": [ "DRUGBANK:DB00394" ]
                },
                "n1": {
                    "categories": [ "biolink:Drug" ]
                }
            }
        }
    },
    "query_options": { "n_results": 5 }
}
```
""",
    response_model=TrapiQuery,
    tags=["reasoner"],
)
def post_reasoner_predict(
        request_body: TrapiQuery = Body(..., example=TRAPI_EXAMPLE)
    ) -> TrapiQuery:
    """Get predicted associations for a given ReasonerAPI query.

    :param request_body: The ReasonerStdAPI query in JSON
    :return: Predictions as a ReasonerStdAPI Message
    """
    query_graph = request_body.message.query_graph.dict(exclude_none=True)
    
    if len(query_graph["edges"]) == 0:
        return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
        # return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)

    if len(query_graph["edges"]) > 1:
        # Currently just return a empty result if multi-edges query
        return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
        # return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)


    # reasonerapi_response = resolve_trapi_query(request_body.dict(exclude_none=True), app=app)
    reasonerapi_response = resolve_trapi_query(
        request_body.dict(exclude_none=True),
        app
    )

    return JSONResponse(reasonerapi_response) or ('Not found', 404)



@app.get("/meta_knowledge_graph", name="Get the meta knowledge graph",
    description="Get the meta knowledge graph",
    response_model=dict,
    tags=["trapi"],
)
def get_meta_knowledge_graph() -> dict:
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
                # TODO: https://github.com/NCATSTranslator/ReasonerAPI/pull/331/files
                # "knowledge_types": ['inferred', 'lookup']
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



@app.get("/predict", name="Get predicted targets for a given entity",
    description="""Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.
Only a drug_id or a disease_id can be provided, the disease_id will be ignored if drug_id is provided
This operation is annotated with x-bte-kgs-operations, and follow the BioThings API recommendations.

You can try:

| disease_id: `OMIM:246300` | drug_id: `DRUGBANK:DB00394` |
| ------- | ---- |
| to check the drug predictions for a disease   | to check the disease predictions for a drug |
""",
    response_model=dict,
    tags=["biothings"],
)
def get_predict(
        drug_id: Optional[str] = None, 
        disease_id: Optional[str] = 'OMIM:246300', 
        model_id: str ='openpredict-baseline-omim-drugbank', 
        min_score: float = None, max_score: float = None, n_results: int = None
    ) -> dict:
    """Get predicted associations for a given entity CURIE.

    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()

    # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
    concept_id = ''
    if drug_id:
        concept_id = drug_id
    elif disease_id:
        concept_id = disease_id
    else:
        return ('Bad request: provide a drugid or diseaseid', 400)

    try:
        prediction_json, source_target_predictions = get_predictions(
            concept_id, model_id, app, min_score, max_score, n_results
        )
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}



@app.get("/similarity", name="Get similar entities",
    description="""Get similar entites for a given entity CURIE.
    
You can try:

| drug_id: `DRUGBANK:DB00394` | disease_id: `OMIM:246300` |
| ------- | ---- |
| model_id: `drugs_fp_embed.txt` | model_id: `disease_hp_embed.txt` |
| to check the drugs similar to a given drug | to check the diseases similar to a given disease   |
""",
    response_model=dict,
    tags=["openpredict"],
)
def get_similarity(
        types: SimilarityTypes ='Diseases', 
        drug_id: Optional[str] = None, 
        disease_id: Optional[str] = 'OMIM:246300', 
        model_id: str ='disease_hp_embed.txt', 
        min_score: float =None, max_score: float =None, n_results: int =None
    ) -> dict:
    """Get similar entites for a given entity CURIE.

    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now()
    if type(types) is SimilarityTypes:
        types = types.value

    # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
    concept_id = ''
    if drug_id:
        concept_id = drug_id
    elif disease_id:
        concept_id = disease_id
    else:
        return ('Bad request: provide a drugid or diseaseid', 400)

    try:
        emb_vectors = app.similarity_embeddings[model_id]
        prediction_json, source_target_predictions = get_similarities(
            types, concept_id, emb_vectors, min_score, max_score, n_results
        )
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    # relation = "biolink:treated_by"
    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}
    # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)


@app.get("/features", name="Return the features trained in the models",
    description="""Return the features trained in the model, for Drugs, Diseases or Both.""",
    response_model=dict,
    tags=["openpredict"],
)
def get_features(embedding_type: EmbeddingTypes ='Drugs') -> dict:
    """Get features in the model

    :return: JSON with features
    """
    if type(embedding_type) is EmbeddingTypes:
        embedding_type = embedding_type.value
    return retrieve_features(embedding_type)



@app.get("/models", name="Return the models with their training features and scores",
    description="""Return the models with their training features and scores""",
    response_model=dict,
    tags=["openpredict"],
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models()



@app.post("/embedding", name="Upload your embedding for drugs or diseases",
    description="""Upload your embedding file:  

1. Select which types do you have in the embeddings: Drugs, Diseases or Both. 

2. Define the base `model_id`: use the `/models` call to see the list of trained models with their characteristics, and pick the ID of the model you will use as base to add your embedding

3. The model will be retrained and evaluation will be stored in a triplestore (available in `/models`)
""",
    response_model=dict,
    tags=["openpredict"],
)
def post_embedding(
        emb_name: str, description: str, 
        types: EmbeddingTypes ='Both', model_id: str ='openpredict-baseline-omim-drugbank', 
        apikey: str=None,
        uploaded_file: UploadFile = File(...)
    ) -> dict:
    """Post JSON embeddings via the API, with simple APIKEY authentication 
    provided in environment variables 
    """
    if type(types) is EmbeddingTypes:
        types = types.value

    # Ignore the API key check if no env variable defined (for development)
    if os.getenv('OPENPREDICT_APIKEY') == apikey or os.getenv('OPENPREDICT_APIKEY') is None:
        embedding_file = uploaded_file.file
        print(emb_name, types)
        run_id, scores = addEmbedding(
            embedding_file, emb_name, types, description, model_id)
        print('Embeddings uploaded')
        # train_model(False)
        return {
            'status': 200,
            'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(scores)
        }
    else:
        return {'Forbidden': 403}


@app.get("/evidence_path", name="Get the evidence path between two entities",
    description="""Get the evidence path between two entities. The evidence path is generated using the overall similarity score by default. You could change the 
    included features by defining the names of the features.
    
You can try:

| drug_id: `DRUGBANK:DB00915` | disease_id: `OMIM:104300` |
| ------- | ---- |
| min_similarity_threshold_drugs/disease : `0.1` | features_drug: `PPI-SIM` | features_disease : `HPO-SIM` |
| (Between 0-1) to include the drugs/diseases which have similarity below the threshold | to select a specific similarity feature for drugs   | to select a specific similarity feature for diseases | 
""",
    response_model=dict,
    tags=["openpredict"],
)
def get_evidence_path(
        drug_id: str = Query(default=..., example="DRUGBANK:DB00915"),
        disease_id: str = Query(default=..., example="OMIM:104300"),
        min_similarity_threshold_drugs: float = 1.0,
        min_similarity_threshold_disease : float = 1.0, 
        features_drug: FeatureTypesDrugs = None,
        features_disease : FeatureTypesDiseases = None

        # model_id: str ='disease_hp_embed.txt', 
        
    ) -> dict:
    """Get similar entites for a given entity CURIE.

    :param entity: Search for predicted associations for this entity CURIE
    :return: Prediction results object with score
    """
    time_start = datetime.now() 

    try:
        # if features_of_interest is not None: 
        #     features_of_interest.upper()
        #     features_of_interest = features_of_interest.split(", ")
        #     path_json = do_evidence_path(drug_id, disease_id,top_K,features_drug, features_disease)
        # else:
        drug_id = drug_id[-7:]
        disease_id = disease_id[-6:]
        # if features_drug is not None : 
        #     features_drug = features_drug.split(", ")
        # if features_disease is not None: 
        #     features_disease = features_disease.split(", ")


        path_json = do_evidence_path(drug_id, disease_id, min_similarity_threshold_drugs,min_similarity_threshold_disease, features_drug, features_disease)
    except Exception as e:
        print(f'Error getting evidence path between {drug_id} and {disease_id}')
        print(e)
        return (f'Evidence path between {drug_id} and {disease_id} not found', 404)
    
    # relation = "biolink:treated_by"
    print('EvidencePathRuntime: ' + str(datetime.now() - time_start))
    return {"output" : path_json,'count': len(path_json)}
    # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)


@app.get("/health", include_in_schema=False)
def health_check():
    """Health check for Translator elastic load balancer"""
    return {'status': 'ok'}


@app.get("/", include_in_schema=False)
def redirect_root_to_docs():
    """Redirect the route / to /docs"""
    return RedirectResponse(url='/docs')


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# async def async_reasoner_predict(request_body):
#     """Get predicted associations for a given ReasonerAPI query.

#     :param request_body: The ReasonerStdAPI query in JSON
#     :return: Predictions as a ReasonerStdAPI Message
#     """
#     return post_reasoner_predict(request_body)

# # TODO: get_predict wrapped in ReasonerStdApi
