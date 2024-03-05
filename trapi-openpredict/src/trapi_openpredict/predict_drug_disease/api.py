from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile
from rdflib import Graph

from trapi_openpredict.predict_drug_disease.evidence_path.predict import do_evidence_path
from trapi_openpredict.predict_drug_disease.utils import retrieve_features, retrieve_models

# from openpredict_model.train import add_embedding

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
models_g.parse("../models/openpredict_baseline.ttl")


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
        print("path_json!!!!", path_json)
    except Exception as e:
        print(f'Error getting evidence path between {drug_id} and {disease_id}')
        print(e)
        return (f'Evidence path between {drug_id} and {disease_id} not found', 404)

    # relation = "biolink:treated_by"
    print('EvidencePathRuntime: ' + str(datetime.now() - time_start))
    return {"output" : path_json,'count': len(path_json)}
    # return {'results': prediction_json, 'relation': relation, 'count': len(prediction_json)} or ('Not found', 404)


@api.get("/features", name="Return the features trained in the models",
    description="""Return the features trained in the model, for Drugs, Diseases or Both.""",
    response_model=dict,
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
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models(models_g)
