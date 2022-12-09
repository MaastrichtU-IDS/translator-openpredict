import os

from gensim.models import KeyedVectors
from joblib import load

from openpredict.config import settings


def load_similarity_embeddings():
    """Load embeddings model for similarity"""
    embedding_folder = os.path.join(settings.OPENPREDICT_DATA_DIR, 'embedding')
    similarity_embeddings = {}
    for model_id in os.listdir(embedding_folder):
        if model_id.endswith('txt'):
            feature_path = os.path.join(embedding_folder, model_id)
            print("📥 Loading similarity features from " + feature_path)
            emb_vectors = KeyedVectors.load_word2vec_format(feature_path)
            similarity_embeddings[model_id]= emb_vectors
    return similarity_embeddings


def load_treatment_classifier(model_id):
    """Load embeddings model for treats and treated_by"""
    print("📥 Loading treatment classifier from joblib for model " + str(model_id))
    model_path = os.path.join(settings.OPENPREDICT_DATA_DIR, "models", f"{str(model_id)}.joblib")
    return load(model_path)


def load_treatment_embeddings(model_id):
    """Load embeddings model for treats and treated_by"""
    print(f"📥 Loading treatment features for model {str(model_id)}")
    model_path = os.path.join(settings.OPENPREDICT_DATA_DIR, "features", f"{str(model_id)}.joblib")
    (drug_df, disease_df) = load(model_path)
    return (drug_df, disease_df)
