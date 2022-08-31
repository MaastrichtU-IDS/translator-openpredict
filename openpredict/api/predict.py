from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from openpredict.config import PreloadedModels
from openpredict.models.drugrepurposing import get_drugrepositioning_results
from openpredict.models.explain import get_explanations
from openpredict.models.openpredict_model import get_predictions, get_similarities
from openpredict.openapi import SimilarityTypes

app = APIRouter()


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
            concept_id, model_id, min_score, max_score, n_results
        )
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}



@app.get("/drugrepositioning", name="Get  predicted drugs for a given disease",
    description="""Return the predicted drugs for a given disease (such as MESHID or OMIMID), with confidence scores.
Only disease_id can be provided, the disease_id will be ignored if drug_id is provided
This operation is annotated with x-bte-kgs-operations, and follow the BioThings API recommendations.

You can try:

| disease_id: `MESH:D000544` | 

| to check the drug prediction explanations  for a disease  |
""",
    response_model=dict,
    tags=["openpredict"],
)

def get_drugrepositioning(
        #drug_id: Optional[str] = None, 
        disease_id: Optional[str] = 'MESH:D000544', 
        #model_id: str ='openpredict-baseline-omim-drugbank', 
        n_results: int = 100
    ) -> dict:
    """Get drug repositioning predictions for a given entity CURIE disease  .

    :param entity: Get predictions associations for this entity CURIE
    :return: Prediction results 
    """
    time_start = datetime.now()
    #return ('test: provide a drugid or diseaseid', 400)
    # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
    concept_id = ''

    if disease_id:
        concept_id = disease_id
    else:
        return ('Bad request: provide a drugid or diseaseid', 400)

    try:
        print('concept' + concept_id)
        prediction_json, source_target_predictions = get_drugrepositioning_results(
            concept_id , n_results
        )
        
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(str(e) )
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
        emb_vectors = PreloadedModels.similarity_embeddings[model_id]
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



@app.get("/explain", name="Get calculated shap explanations for  predicted drug for a given disease",
    description="""Return the explanations for predicted entities  for a given disease  with SHAP values for feature importances: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.
a disease_id can be provided,
This operation is annotated with x-bte-kgs-operations, and follow the BioThings API recommendations.

You can try:

| disease_id: `OMIM:246300` | 

| to check the drug prediction explanations  for a disease  |
""",
    response_model=dict,
    tags=["openpredict"],
)
def get_explanation(
        #drug_id: Optional[str] = None, 
        disease_id: Optional[str] = 'OMIM:246300', 
        #model_id: str ='openpredict-baseline-omim-drugbank', 
        n_results: int = 100
    ) -> dict:
    """Get explanations for a given entity CURIE disease and predicted drugs.

    :param entity: Get explanations associations for this entity CURIE
    :return: Prediction results with shap values for all features  in the  ML model with score
    """
    time_start = datetime.now()
    #return ('test: provide a drugid or diseaseid', 400)
    # TODO: if drug_id and disease_id defined, then check if the disease appear in the provided drug predictions
    concept_id = ''
    drug_id= None
    model_id=None
    min_score=None
    max_score=None
    if drug_id:
        concept_id = drug_id
    elif disease_id:
        concept_id = disease_id
    else:
        return ('Bad request: provide a drugid or diseaseid', 400)

    try:
        
        prediction_json, source_target_predictions = get_explanations(
            concept_id, model_id, app, min_score, max_score, n_results
        )
        
    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}
import sys
