import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gensim.models import KeyedVectors
from joblib import load
from pydantic import AnyHttpUrl, BaseSettings, EmailStr, HttpUrl, PostgresDsn, validator

from openpredict.config import settings
from openpredict.models.evidence_path_model import EvidencePathModel
from openpredict_model.utils import load_similarity_embeddings, load_treatment_classifier, load_treatment_embeddings


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
