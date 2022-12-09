from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from openpredict_explain_shap.predict import get_explanations

api = APIRouter()


@api.get("/explain", name="Get calculated shap explanations for  predicted drug for a given disease",
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

        prediction_json = get_explanations(
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
