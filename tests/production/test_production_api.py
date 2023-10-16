import os

import requests
from reasoner_validator.validator import TRAPIResponseValidator
from trapi_predict_kit import settings

# NOTE: Validate only prod because validate requires py3.9+ and OpenPredict requires 3.8
validator = TRAPIResponseValidator(
    trapi_version=settings.TRAPI_VERSION,

    # If omit or set the Biolink Model version parameter to None,
    # then the current Biolink Model Toolkit default release applies
    biolink_version=settings.BIOLINK_VERSION,

    # 'sources' are set to trigger checking of expected edge knowledge source provenance
    # sources={
    #     "ara_source": "infores:molepro",
    #     "kp_source": "infores:knowledge-collaboratory",
    #     "kp_source_type": "primary"
    # },
    # Optional flag: if omitted or set to 'None', we let the system decide the
    # default validation strictness by validation context unless we override it here
    strict_validation=None
)


def check_trapi_compliance(response):
    # validator.check_compliance_of_trapi_response(response.json()["message"])
    validator.check_compliance_of_trapi_response(response)
    validator_resp = validator.get_messages()
    print("⚠️ REASONER VALIDATOR WARNINGS:")
    print(validator_resp["warnings"])
    if len(validator_resp["errors"]) == 0:
        print("✅ NO REASONER VALIDATOR ERRORS")
    else:
        print("🧨 REASONER VALIDATOR ERRORS")
        print(validator_resp["errors"])
    assert (
        len(validator_resp["errors"]) == 0
    )


def test_openapi_description(pytestconfig):
    # https://smart-api.info/api/validate?url=https://openpredict.semanticscience.org/openapi.json
    openapi_desc = requests.get(
        "https://smart-api.info/api/validate",
        params={
            'url': pytestconfig.getoption("server") + '/openapi.json',
        },
        timeout=300
    ).json()
    assert openapi_desc["success"]

def test_get_predict(pytestconfig):
    """Test predict API GET operation"""
    print(f'🧪 Testing API: {pytestconfig.getoption("server")}')
    get_predictions = requests.post(
        pytestconfig.getoption("server") + '/predict',
        json={
            "subjects": ["DRUGBANK:DB00394"],
            "options": {
                "model_id": "openpredict_baseline",
                "n_results": 42,
        }},
        timeout=300
    ).json()
    assert 'hits' in get_predictions
    assert len(get_predictions['hits']) == 42
    assert get_predictions['count'] == 42
    # assert get_predictions['hits'][0]['id'] == 'OMIM:246300'


# TODO: add tests using a TRAPI validation API if possible?
def test_post_trapi(pytestconfig):
    """Test Translator ReasonerAPI query POST operation to get predictions"""
    headers = {'Content-type': 'application/json'}

    for trapi_filename in os.listdir(os.path.join('tests', 'queries')):

        with open(os.path.join('tests', 'queries', trapi_filename)) as f:
            trapi_query = f.read()
            response = requests.post(
                pytestconfig.getoption("server") + '/query',
                data=trapi_query,
                headers=headers,
                timeout=300, # 5min timeout
            ).json()
            edges = response['message']['knowledge_graph']['edges'].items()

            check_trapi_compliance(response)
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
