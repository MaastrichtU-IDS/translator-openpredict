from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter

from drkg_model.predict import get_drugrepositioning_results
from openpredict.loaded_models import PreloadedModels
from openpredict_model.predict import get_predictions, get_similarities

api = APIRouter()


@api.get("/predict", name="Get predicted targets for a given entity",
    description="""Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.
Only a drug_id or a disease_id can be provided, the disease_id will be ignored if drug_id is provided
This operation is annotated with x-bte-kgs-operations, and follow the BioThings API recommendations.

You can try:

| disease_id: `OMIM:246300` | drug_id: `DRUGBANK:DB00394` |
| ------- | ---- |
| to check the drug predictions for a disease   | to check the disease predictions for a drug |
""",
    response_model=dict,
    tags=["openpredict"],
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
        prediction_json = get_predictions(
            concept_id, model_id, min_score, max_score, n_results
        )
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}



class SimilarityTypes(str, Enum):
    Drugs = "Drugs"
    Diseases = "Diseases"


@api.get("/similarity", name="Get similar entities",
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
        emb_vectors = PreloadedModels.similarity_embeddings[model_id]
        prediction_json = get_similarities(
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

