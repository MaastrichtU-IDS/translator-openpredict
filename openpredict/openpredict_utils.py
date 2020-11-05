import os
import shutil
import logging
import requests
from openpredict.rdf_utils import add_feature_metadata

OPENPREDICT_DATA_FOLDER = os.getenv('OPENPREDICT_DATA_FOLDER')
if not OPENPREDICT_DATA_FOLDER:
    OPENPREDICT_DATA_FOLDER = 'data/'
    # OPENPREDICT_DATA_FOLDER = '/data/openpredict/'

def get_openpredict_dir(subfolder=''):
    """Return the full path to the provided files in the OpenPredict data folder
    Where models and features for runs are stored
    """
    return OPENPREDICT_DATA_FOLDER + subfolder

def init_openpredict_dir():
    """Create OpenPredict folder and initiatite files if necessary.
    Also create baseline features in the triplestore
    """
    if not os.path.exists(get_openpredict_dir()):
        print('Creating ' + get_openpredict_dir())
        os.makedirs(get_openpredict_dir())
    if not os.path.exists(get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')):
        print('Initiating ' + get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib'))
        shutil.copy('data/features/openpredict-baseline-omim-drugbank.joblib', get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib'))
    if not os.path.exists(get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib')):
        print('Initiating ' + get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib'))
        shutil.copy('data/models/openpredict-baseline-omim-drugbank.joblib', get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib'))
    add_feature_metadata("GO-SIM", "GO based drug-drug similarity", "Drugs")
    add_feature_metadata("TARGETSEQ-SIM", "Drug target sequence similarity: calculation of SmithWaterman sequence alignment scores", "Drugs")
    add_feature_metadata("PPI-SIM", "PPI based drug-drug similarity, calculate distance between drugs on protein-protein interaction network", "Drugs")
    add_feature_metadata("TC", "Drug fingerprint similarity, calculating MACS based fingerprint (substructure) similarity", "Drugs")
    add_feature_metadata("SE-SIM", "Drug side effect similarity, calculating Jaccard coefficient based on drug sideefects", "Drugs")
    add_feature_metadata("PHENO-SIM", "Disease Phenotype Similarity based on MESH terms similarity", "Diseases")
    add_feature_metadata("HPO-SIM", "HPO based disease-disease similarity", "Diseases")



def get_entities_labels(entity_list):
    """Send the list of node IDs to Translator Normalization API to get labels
    See API: https://nodenormalization-sri.renci.org/apidocs/#/Interfaces/get_get_normalized_nodes
    and example notebook: https://github.com/TranslatorIIPrototypes/NodeNormalization/blob/master/documentation/NodeNormalization.ipynb
    """
    # TODO: add the preferred identifier CURIE to our answer also?
    try:
        get_label_result = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                            params={'curie': entity_list})
        get_label_result = get_label_result.json()
    except:
        # Catch if the call to the API fails (API not available)
        logging.info("Translator API down: https://nodenormalization-sri.renci.org/apidocs")
        get_label_result = {}
    # Response is a JSON:
    # { "HP:0007354": {
    #     "id": { "identifier": "MONDO:0004976",
    #       "label": "amyotrophic lateral sclerosis" },
    return get_label_result