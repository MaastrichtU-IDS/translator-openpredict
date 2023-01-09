import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile
from rdflib import Graph

from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict_model.evidence_path.predict import do_evidence_path
from openpredict_model.explain_shap.explain_shap import get_explanations
from openpredict_model.train import add_embedding


class FeatureTypesDrugs(str, Enum):
    PPI_SIM = "PPI-SIM"
    TC = "TC"
    SE_SIM = "SE-SIM"
    TARGETSEQ_SIM = "TARGETSEQ-SIM"
    GO_SIM = "GO-SIM"

class FeatureTypesDiseases(str, Enum) :
    HPO_SIM = "HPO-SIM"
    PHENO_SIM = "PHENO-SIM"

class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


api = APIRouter()

models_g = Graph()
models_g.parse("models/openpredict_baseline.ttl")


@api.get("/evidence-path", name="Get the evidence path between two entities",
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

        # Remove namespaces from IDs:
        drug_id = drug_id[-7:]
        disease_id = disease_id[-6:]
        # if features_drug is not None :
        #     features_drug = features_drug.split(", ")
        # if features_disease is not None:
        #     features_disease = features_disease.split(", ")


        path_json = do_evidence_path(
            drug_id, disease_id,
            min_similarity_threshold_drugs, min_similarity_threshold_disease,
            features_drug, features_disease
        )
    except Exception as e:
        print(f'Error getting evidence path between {drug_id} and {disease_id}')
        print(e)
        return (f'Evidence path between {drug_id} and {disease_id} not found', 404)

    # relation = "biolink:treated_by"
    print('EvidencePathRuntime: ' + str(datetime.now() - time_start))
    return {"output" : path_json,'count': len(path_json)}
    # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)



@api.get("/explain-shap", name="Get calculated shap explanations for  predicted drug for a given disease",
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
        #model_id: str ='openpredict_baseline',
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
            concept_id, model_id, min_score, max_score, n_results
        )

    except Exception as e:
        print('Error processing ID ' + concept_id)
        print(e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        return ('Not found: entry in OpenPredict for ID ' + concept_id, 404)

    print('PredictRuntime: ' + str(datetime.now() - time_start))
    return {'hits': prediction_json, 'count': len(prediction_json)}



@api.get("/features", name="Return the features trained in the models",
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
    return retrieve_features(models_g, embedding_type)



@api.get("/models", name="Return the models with their training features and scores",
    description="""Return the models with their training features and scores""",
    response_model=dict,
    tags=["openpredict"],
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models(models_g)



@api.post("/embedding", name="Upload your embedding for drugs or diseases",
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
        types: EmbeddingTypes ='Both', model_id: str ='openpredict_baseline',
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
        run_id, loaded_model = add_embedding(
            embedding_file, emb_name, types, model_id)
        print('Embeddings uploaded')
        # train_model(False)
        return {
            'status': 200,
            'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(loaded_model.scores)
        }
    else:
        return {'Forbidden': 403}
