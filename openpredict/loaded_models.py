import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pkg_resources
from gensim.models import KeyedVectors
from joblib import load
from openpredict.config import settings
from openpredict.ml_models.evidence_path_model import EvidencePathModel
from openpredict.utils import (
    load_similarity_embeddings,
    load_treatment_classifier,
    load_treatment_embeddings,
)
from pydantic import AnyHttpUrl, BaseSettings, EmailStr, HttpUrl, PostgresDsn, validator


# A class to preload some models before starting the API
class PreloadedModels(object):
  baseline_model_treatment: str
  treatment_embeddings = None
  treatment_classifier = None

  baseline_model_similarity: str
  similarity_embeddings = None

  evidence_path: EvidencePathModel = None
  # similarity: SimilarityModel = None
  # explain_shap: ExplainShapModel = None
  # openpredict: OpenPredictModel = None
  # drug_repurposing_kg: DrugRepurposingKGModel = None


  @classmethod
  def init(
    cls,
    baseline_model_treatment: Optional[str] = 'openpredict-baseline-omim-drugbank',
    baseline_model_similarity: Optional[str] = 'openpredict-baseline-omim-drugbank'
  ) -> None:
    print('Loading PreloadedModels')
    cls.baseline_model_treatment = baseline_model_treatment
    cls.baseline_model_similarity = baseline_model_similarity
    # Initialize embeddings features and classifiers to be used by the API
    cls.treatment_embeddings = load_treatment_embeddings(baseline_model_treatment)
    cls.treatment_classifier = load_treatment_classifier(baseline_model_treatment)
    cls.similarity_embeddings = load_similarity_embeddings()

    cls.evidence_path = EvidencePathModel()


# def load_similarity_embeddings():
#     """Load embeddings model for similarity"""
#     embedding_folder = 'data/embedding'
#     # print(pkg_resources.resource_filename('openpredict', embedding_folder))
#     similarity_embeddings = {}
#     for model_id in os.listdir(pkg_resources.resource_filename('openpredict', embedding_folder)):
#         if model_id.endswith('txt'):
#             feature_path = pkg_resources.resource_filename('openpredict', os.path.join(embedding_folder, model_id))
#             print("📥 Loading similarity features from " + feature_path)
#             emb_vectors = KeyedVectors.load_word2vec_format(feature_path)
#             similarity_embeddings[model_id]= emb_vectors
#     return similarity_embeddings


# def load_treatment_classifier(model_id):
#     """Load embeddings model for treats and treated_by"""
#     print("📥 Loading treatment classifier from joblib for model " + str(model_id))
#     return load(f'{settings.OPENPREDICT_DATA_DIR}/models/{str(model_id)}.joblib')


# def load_treatment_embeddings(model_id):
#     """Load embeddings model for treats and treated_by"""
#     print(f"📥 Loading treatment features for model {str(model_id)}")
#     (drug_df, disease_df) = load(f'{settings.OPENPREDICT_DATA_DIR}/features/{str(model_id)}.joblib')
#     return (drug_df, disease_df)

