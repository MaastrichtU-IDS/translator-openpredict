import os

from gensim.models import KeyedVectors
from joblib import load

from openpredict.config import settings
from openpredict.utils import log

default_model_id = "openpredict-baseline-omim-drugbank"
treatment_embeddings = load(os.path.join(settings.OPENPREDICT_DATA_DIR, "features", f"{default_model_id}.joblib"))
treatment_classifier = load(os.path.join(settings.OPENPREDICT_DATA_DIR, "models", f"{default_model_id}.joblib"))

# Preload similarity embeddings
embedding_folder = os.path.join(settings.OPENPREDICT_DATA_DIR, 'embedding')
similarity_embeddings = {}
for model_id in os.listdir(embedding_folder):
    if model_id.endswith('txt'):
        feature_path = os.path.join(embedding_folder, model_id)
        log.info(f"📥 Loading similarity features from {feature_path}")
        emb_vectors = KeyedVectors.load_word2vec_format(feature_path)
        similarity_embeddings[model_id]= emb_vectors


def load_treatment_classifier(model_id: str = default_model_id):
    """Load embeddings model for treats and treated_by"""
    print("📥 Loading treatment classifier from joblib for model " + str(model_id))
    if (model_id == default_model_id):
        return treatment_classifier

    return load(os.path.join(settings.OPENPREDICT_DATA_DIR, "models", f"{model_id}.joblib"))


def load_treatment_embeddings(model_id: str = default_model_id):
    """Load embeddings model for treats and treated_by"""
    print(f"📥 Loading treatment features for model {str(model_id)}")
    if (model_id == default_model_id):
        return treatment_embeddings

    return load(os.path.join(settings.OPENPREDICT_DATA_DIR, "features", f"{model_id}.joblib"))


def load_similarity_embeddings():
    """Load embeddings model for similarity"""
    return similarity_embeddings
