import os
import pickle

import pandas as pd
from gensim.models import KeyedVectors

from openpredict.config import settings
from openpredict.utils import log, normalize_id_to_translator

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


# TODO: not used
MISSING_IDS = set()
def convert_baseline_features_ids():
    """Convert IDs to use Translator preferred IDs when building the baseline model from scratch"""
    baseline_features_folder = os.path.join(settings.OPENPREDICT_DATA_DIR, "baseline_features")
    drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv',
                    'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    drugfeatfiles = [
        os.path.join(baseline_features_folder, fn) for fn in drugfeatfiles
    ]
    diseasefeatfiles = [
        os.path.join(baseline_features_folder, fn) for fn in diseasefeatfiles
    ]

    # Prepare drug-disease dictionary
    drugDiseaseKnown = pd.read_csv(os.path.join(settings.OPENPREDICT_DATA_DIR, 'resources', 'openpredict-omim-drug.csv'),delimiter=',')
    drugDiseaseKnown.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)

    drugs_set = set()
    diseases_set = set()
    drugs_set.update(drugDiseaseKnown['Drug'].tolist())
    diseases_set.update(drugDiseaseKnown['Disease'].tolist())

    for csv_file in drugfeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        drugs_set.update(df['Drug1'].tolist())
        drugs_set.update(df['Drug2'].tolist())

    for csv_file in diseasefeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        diseases_set.update(df['Disease1'].tolist())
        diseases_set.update(df['Disease2'].tolist())

    diseases_set = ['OMIM:{}'.format(disease) for disease in diseases_set]
    drugs_set = ['DRUGBANK:{}'.format(drug) for drug in drugs_set]

    diseases_mappings = normalize_id_to_translator(diseases_set)
    drugs_mappings = normalize_id_to_translator(drugs_set)

    log.info('Finished API queries')
    # Replace Ids with translator IDs in kown drug disease associations
    drugDiseaseKnown["Drug"] = drugDiseaseKnown["Drug"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row)     )
    drugDiseaseKnown["Disease"] = drugDiseaseKnown["Disease"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
    drugDiseaseKnown.to_csv('openpredict/data/resources/known-drug-diseases.csv', index=False)

    # Replace IDs in drugs baseline features files
    for csv_file in drugfeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        df["Drug1"] = df["Drug1"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row) )
        df["Drug2"] = df["Drug2"].apply (lambda row: map_id_to_translator(drugs_mappings, 'DRUGBANK:' + row) )
        df.to_csv(csv_file.replace('/baseline_features/', '/translator_features/'), index=False)

    # Replace IDs in diseases baseline features files
    for csv_file in diseasefeatfiles:
        df = pd.read_csv(csv_file, delimiter=',')
        df["Disease1"] = df["Disease1"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
        df["Disease2"] = df["Disease2"].apply (lambda row: map_id_to_translator(diseases_mappings, 'OMIM:' + str(row)) )
        df.to_csv(csv_file.replace('/baseline_features/', '/translator_features/'), index=False)

    log.warn(f"❌️ Missing IDs: {', '.join(MISSING_IDS)}")

    # drugs_set.add(2)
    # drugs_set.update([2, 3, 4])
    # Extract the dataframes col1 and 2 to a unique list
    # Add those list to the drugs and diseases sets
    # Convert the set/list it using normalize_id_to_translator(ids_list)
    # Update all dataframes using the created mappings
    # And store to baseline_translator


def map_id_to_translator(mapping_obj, source_id):
    try:
        return mapping_obj[source_id]
    except:
        MISSING_IDS.add(source_id)
        return source_id
