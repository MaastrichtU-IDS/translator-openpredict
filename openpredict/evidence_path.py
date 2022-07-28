import pkg_resources

from openpredict.utils import get_openpredict_dir

# Access uncommitted data in the persistent data directory
# get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')

# Access the openpredict/data folder for data that has been committed
# pkg_resources.resource_filename('openpredict', 'data/features/openpredict-baseline-omim-drugbank.joblib')

def do_evidence_path(drug_id: str, disease_id: str):
    # Do your thing

    return {
        'drug': drug_id,
        'disease': disease_id
    }