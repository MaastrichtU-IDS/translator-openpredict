from fastapi.responses import RedirectResponse

from openpredict.api import models, predict, trapi
from openpredict.config import PreloadedModels
from openpredict.openapi import TRAPI
from openpredict.utils import init_openpredict_dir

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

init_openpredict_dir()
PreloadedModels.init()

app = TRAPI(
    baseline_model_treatment='openpredict-baseline-omim-drugbank',
    baseline_model_similarity='drugs_fp_embed.txt',
)

app.include_router(trapi.app)
app.include_router(predict.app)
app.include_router(models.app)



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
