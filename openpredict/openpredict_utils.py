import os
import shutil
import logging
import requests
import pkg_resources
from openpredict.rdf_utils import init_triplestore

global OPENPREDICT_DATA_DIR
OPENPREDICT_DATA_DIR = os.getenv('OPENPREDICT_DATA_DIR')
if not OPENPREDICT_DATA_DIR:
    # Data folder in current dir if not provided via environment variable
    OPENPREDICT_DATA_DIR = os.getcwd() + '/data/'
else:
    if not OPENPREDICT_DATA_DIR.endswith('/'):
        OPENPREDICT_DATA_DIR += '/'


def get_openpredict_dir(subfolder=''):
    """Return the full path to the provided files in the OpenPredict data folder
    Where models and features for runs are stored
    """
    return OPENPREDICT_DATA_DIR + subfolder

def init_openpredict_dir():
    """Create OpenPredict folder and initiate files if necessary.
    Also create baseline features in the triplestore
    """
    print('Using directory: ' + OPENPREDICT_DATA_DIR)
    if not os.path.exists(get_openpredict_dir()):
        print('Creating ' + get_openpredict_dir())
        os.makedirs(get_openpredict_dir())
    if not os.path.exists(get_openpredict_dir('features')):
        print('Creating ' + get_openpredict_dir('features'))
        os.makedirs(get_openpredict_dir('features'))
    if not os.path.exists(get_openpredict_dir('models')):
        print('Creating ' + get_openpredict_dir('models'))
        os.makedirs(get_openpredict_dir('models'))
    if not os.path.exists(get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')):
        print('Initiating ' + get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib'))
        shutil.copy(pkg_resources.resource_filename('openpredict', 'data/features/openpredict-baseline-omim-drugbank.joblib'),
            get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib'))
    if not os.path.exists(get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib')):
        print('Initiating ' + get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib'))
        shutil.copy(pkg_resources.resource_filename('openpredict', 'data/models/openpredict-baseline-omim-drugbank.joblib'), 
            get_openpredict_dir('models/openpredict-baseline-omim-drugbank.joblib'))
    if not os.path.exists(get_openpredict_dir('openpredict-metadata.ttl')):
        print('Creating ' + get_openpredict_dir('openpredict-metadata.ttl'))
        # shutil.copy(get_openpredict_dir('initial-openpredict-metadata.ttl'), 
        shutil.copy(pkg_resources.resource_filename('openpredict', 'data/openpredict-metadata.ttl'), 
            get_openpredict_dir('openpredict-metadata.ttl'))
    init_triplestore()
    # Check if https://w3id.org/openpredict/run/openpredict-baseline-omim-drugbank exist before iniating the triplestore
    # add_feature_metadata("GO-SIM", "GO based drug-drug similarity", "Drugs")
    # add_feature_metadata("TARGETSEQ-SIM", "Drug target sequence similarity: calculation of SmithWaterman sequence alignment scores", "Drugs")
    # add_feature_metadata("PPI-SIM", "PPI based drug-drug similarity, calculate distance between drugs on protein-protein interaction network", "Drugs")
    # add_feature_metadata("TC", "Drug fingerprint similarity, calculating MACS based fingerprint (substructure) similarity", "Drugs")
    # add_feature_metadata("SE-SIM", "Drug side effect similarity, calculating Jaccard coefficient based on drug sideefects", "Drugs")
    # add_feature_metadata("PHENO-SIM", "Disease Phenotype Similarity based on MESH terms similarity", "Diseases")
    # add_feature_metadata("HPO-SIM", "HPO based disease-disease similarity", "Diseases")



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