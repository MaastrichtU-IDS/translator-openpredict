import logging
import os
from itertools import zip_longest

import pandas as pd
import requests

from openpredict.config import settings

MISSING_IDS = set()

## Instantiate logging utility
log = logging.getLogger("uvicorn.error")
log.propagate = False
log_level = logging.ERROR
if settings.DEV_MODE:
    log_level = logging.INFO
log.setLevel(log_level)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: [%(module)s:%(funcName)s] %(message)s"
)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)



def get_openpredict_dir(subfolder=''):
    """Return the full path to the provided files in the OpenPredict data folder
    Where models and features for runs are stored
    """
    if not settings.OPENPREDICT_DATA_DIR.endswith('/'):
        settings.OPENPREDICT_DATA_DIR += '/'
    return settings.OPENPREDICT_DATA_DIR + subfolder


def init_openpredict_dir():
    """Create OpenPredict folder and initiate files if necessary."""
    if not os.path.exists(get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')):
        raise ValueError("❌ The data required to run the prediction models could not be found in the `data` folder"
            "ℹ️ Use `pip install dvc` and `dvc pull` to pull the data easily")


def split_list(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_entities_labels(entity_list):
    """Send the list of node IDs to Translator Normalization API to get labels
    See API: https://nodenormalization-sri.renci.org/apidocs/#/Interfaces/get_get_normalized_nodes
    and example notebook: https://github.com/TranslatorIIPrototypes/NodeNormalization/blob/master/documentation/NodeNormalization.ipynb
    """
    # TODO: add the preferred identifier CURIE to our answer also?
    label_results = {}
    entity_list = list(entity_list)
    for chunk in split_list(entity_list, 300):
        try:
            get_label_result = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                                params={'curie': chunk})
            label_results.update(get_label_result.json())
            # print(f"get_entities_labels {get_label_result}")
        except Exception as e:
            # Catch if the call to the API fails (API not available)
            logging.warn(f"Error getting entities labels from NodeNormalization API ({e}), it might be down: https://nodenormalization-sri.renci.org/docs")
    return label_results


def normalize_id_to_translator(ids_list):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID https://nodenormalization-sri.renci.org/docs
    """
    converted_ids_obj = {}
    resolve_curies = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                    params={'curie': ids_list})
    # Get corresponding OMIM IDs for MONDO IDs if match
    resp = resolve_curies.json()
    # print(resp)
    for converted_id, translator_ids in resp.items():
        try:
            pref_id = translator_ids['id']['identifier']
            log.info(converted_id + ' > ' + pref_id)
            converted_ids_obj[converted_id] = pref_id
        except:
            log.error('❌️ ' + converted_id + ' > ' + str(translator_ids))

    return converted_ids_obj

def get_entity_types(entity):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID https://nodenormalization-sri.renci.org/docs
    """
    resolve_curies = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                    params={'curie': [entity]})
    # Get corresponding OMIM IDs for MONDO IDs if match
    resp = resolve_curies.json()
    if entity in resp:
        return resp[entity]["type"]
    return []


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
