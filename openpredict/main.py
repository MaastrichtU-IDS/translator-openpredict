import os
from datetime import datetime
from openpredict.utils import init_openpredict_dir
from openpredict.rdf_utils import init_triplestore, retrieve_features, retrieve_models
from openpredict.openpredict_model import addEmbedding, get_predictions, get_similarities, load_similarity_embeddings
from openpredict.trapi_parser import resolve_trapi_query
from openpredict.openapi import TRAPI, TRAPI_EXAMPLE, EmbeddingTypes, SimilarityTypes

# from gensim.models import KeyedVectors
# import asyncio
# import aiohttp
# from aiohttp import web
# import logging

from fastapi import FastAPI, Body, Request, Response, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from reasoner_pydantic import Query, Message
from typing import Optional, Dict


init_openpredict_dir()
init_triplestore()

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
    response_model=Query,
    tags=["reasoner"],
)
def post_reasoner_predict(
        request_body: Query = Body(..., example=TRAPI_EXAMPLE)
    ) -> Query:
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
        drug_id: Optional[str] =None, 
        disease_id: Optional[str] =None, 
        model_id: str ='openpredict-baseline-omim-drugbank', 
        min_score: float =None, max_score: float =None, n_results: int =None
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
        types: SimilarityTypes ='Drugs', 
        drug_id: Optional[str] =None, 
        disease_id: Optional[str] =None, 
        model_id: str ='drugs_fp_embed.txt', 
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
    description="""Upload your embedding file:  select which types do you have in the embeddings, Drugs, Diseases or Both. 
        1. provided embeddings will be added to the model
        2. the model will be retrained
        3. the model evaluation will be stored in a triplestore""",
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
