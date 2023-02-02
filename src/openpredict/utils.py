import logging
import os
from itertools import zip_longest

import requests

from openpredict.config import settings

## Instantiate logging utility
log = logging.getLogger("uvicorn.error")
log.propagate = False
log_level = logging.getLevelName(settings.LOG_LEVEL)
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
    if not os.path.exists(get_openpredict_dir('input/drugbank-drug-goa.csv')):
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
