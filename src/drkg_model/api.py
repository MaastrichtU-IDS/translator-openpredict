from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from drkg_model.predict import get_drugrepositioning_results

api = APIRouter()


@api.get("/drugrepositioning", name="Get  predicted drugs for a given disease",
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
        prediction_json = get_drugrepositioning_results(
            concept_id , n_results
        )

    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(str(e) )
        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}
