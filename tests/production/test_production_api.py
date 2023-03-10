import os

import requests
from reasoner_validator import TRAPIResponseValidator

from openpredict.config import settings


PROD_API_URL = 'https://openpredict.semanticscience.org'
# PROD_API_URL = 'https://openpredict.137.120.31.148.sslip.io'


# NOTE: Validate only prod because validate requires py3.9+ and OpenPredict requires 3.8
validator = TRAPIResponseValidator(
    trapi_version=settings.TRAPI_VERSION,

    # If omit or set the Biolink Model version parameter to None,
    # then the current Biolink Model Toolkit default release applies
    biolink_version=settings.BIOLINK_VERSION,

    # 'sources' are set to trigger checking of expected edge knowledge source provenance
    sources={
            # "ara_source": "infores:molepro",
            # "kp_source": "infores:knowledge-collaboratory",
            # "kp_source_type": "primary"
    },
    # Optional flag: if omitted or set to 'None', we let the system decide the
    # default validation strictness by validation context unless we override it here
    strict_validation=None
)



def test_get_predict():
    """Test predict API GET operation"""
    # url = PROD_API_URL + '/predict?drug_id=DRUGBANK:DB00394&model_id=openpredict_baseline&n_results=42'
    get_predictions = requests.get(
        PROD_API_URL + '/predict',
        params={
            'input_id': 'DRUGBANK:DB00394',
            'n_results': '42',
            'model_id': 'openpredict_baseline'
        }
    ).json()
    assert 'hits' in get_predictions
    assert len(get_predictions['hits']) == 42
    assert get_predictions['count'] == 42
    # assert get_predictions['hits'][0]['id'] == 'OMIM:246300'


# TODO: add tests using a TRAPI validation API if possible?
def test_post_trapi():
    """Test Translator ReasonerAPI query POST operation to get predictions"""
    headers = {'Content-type': 'application/json'}

    for trapi_filename in os.listdir(os.path.join('tests', 'queries')):

        with open(os.path.join('tests', 'queries', trapi_filename)) as f:
            trapi_query = f.read()
            trapi_results = requests.post(PROD_API_URL + '/query',
                        data=trapi_query, headers=headers).json()
            edges = trapi_results['message']['knowledge_graph']['edges'].items()

            print(trapi_filename)
            validator.check_compliance_of_trapi_response(message=trapi_results["message"])
            validator_resp = validator.get_messages()
            print(validator_resp["warnings"])
            assert (
                len(validator_resp["errors"]) == 0
            )
            if trapi_filename.endswith('0.json'):
                assert len(edges) == 0
            elif trapi_filename.endswith('limit3.json'):
                assert len(edges) == 3
            elif trapi_filename.endswith('limit1.json'):
                assert len(edges) == 1
            else:
                assert len(edges) >= 5



# TODO: Check for this edge structure:
#   "knowledge_graph": {
#     "edges": {
#       "e0": {
#         "attributes": [
#           {
#             "name": "model_id",
#             "source": "OpenPredict",
#             "type": "EDAM:data_1048",
#             "value": "openpredict_baseline"
#           },
#           {
#             "name": "score",
#             "source": "OpenPredict",
#             "type": "EDAM:data_1772",
#             "value": "0.8267106697312154"
#           }
#         ],
#         "object": "DRUGBANK:DB00394",
#         "predicate": "biolink:treated_by",
#         "relation": "RO:0002434",
#         "subject": "OMIM:246300"
#       },
