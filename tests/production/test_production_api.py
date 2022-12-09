import os

import requests

from tests.conftest import validator

PROD_API_URL = 'https://openpredict.semanticscience.org'
# PROD_API_URL = 'https://openpredict.137.120.31.148.sslip.io'


def test_get_predict():
    """Test predict API GET operation"""
    # url = PROD_API_URL + '/predict?drug_id=DRUGBANK:DB00394&model_id=openpredict-baseline-omim-drugbank&n_results=42'
    get_predictions = requests.get(PROD_API_URL + '/predict',
                        params={
                            'drug_id': 'DRUGBANK:DB00394',
                            'n_results': '42',
                            'model_id': 'openpredict-baseline-omim-drugbank'
                        }).json()
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
            if trapi_filename.endswith('limit3.json'):
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
#             "value": "openpredict-baseline-omim-drugbank"
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
