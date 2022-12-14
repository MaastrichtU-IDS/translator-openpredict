import os
import pickle

from gensim.models import KeyedVectors

from openpredict.config import settings
from openpredict.utils import log

default_model_id = "openpredict_baseline"
features_embeddings = pickle.load(open(f"data/features/{default_model_id}_features.pickle", "rb"))
# features_embeddings = pickle.load(open(os.path.join(settings.OPENPREDICT_DATA_DIR, "features", f"{default_model_id}_features.pickle"), "rb"))

# Preload similarity embeddings
embedding_folder = os.path.join(settings.OPENPREDICT_DATA_DIR, 'embedding')
similarity_embeddings = {}
for model_id in os.listdir(embedding_folder):
    if model_id.endswith('txt'):
        feature_path = os.path.join(embedding_folder, model_id)
        log.info(f"📥 Loading similarity features from {feature_path}")
        emb_vectors = KeyedVectors.load_word2vec_format(feature_path)
        similarity_embeddings[model_id]= emb_vectors


def load_features_embeddings(model_id: str = default_model_id):
    """Load embeddings model for treats and treated_by"""
    print(f"📥 Loading treatment features for model {str(model_id)}")
    if (model_id == default_model_id):
        return features_embeddings

    return pickle.load(open(f"data/features/{model_id}_features.pickle", "rb"))


def load_similarity_embeddings():
    """Load embeddings model for similarity"""
    return similarity_embeddings
