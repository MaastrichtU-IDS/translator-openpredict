import pytest
import pkg_resources
import connexion
import os
import json
from openpredict.openpredict_utils import init_openpredict_dir
from reasoner_validator import validate

# Create and start Flask from openapi.yml before running tests
init_openpredict_dir()
flask_app = connexion.FlaskApp(__name__)
flask_app.add_api('../../openpredict/openapi.yml')
@pytest.fixture(scope='module')
def client():
    with flask_app.app.test_client() as c:
        yield c

def test_get_predict(client):
    """Test predict API GET operation"""
    url = '/predict?drug_id=DRUGBANK:DB00394&model_id=openpredict-baseline-omim-drugbank&n_results=42'
    response = client.get(url)
    assert len(response.json['hits']) == 42
    assert response.json['count'] == 42
    assert response.json['hits'][0]['id'] == 'OMIM:246300'

def test_post_trapi(client):
    """Test Translator ReasonerAPI query POST operation to get predictions"""
    url = '/query'
    for trapi_filename in os.listdir(pkg_resources.resource_filename('tests', 'queries')):
        with open(pkg_resources.resource_filename('tests', 'queries/' + trapi_filename),'r') as f:
            reasoner_query = f.read()
            response = client.post(url, 
                                    data=reasoner_query, 
                                    content_type='application/json')

            # print(response.json)
            edges = response.json['message']['knowledge_graph']['edges'].items()
            # print(response)
            print(trapi_filename)
            assert validate(response.json['message'], "Message", "1.1.0") == None
            if trapi_filename.endswith('limit3.json'):
                assert len(edges) == 3
            elif trapi_filename.endswith('limit1.json'):
                assert len(edges) == 1
            else:
                assert len(edges) >= 5


def test_trapi_empty_response(client):
    reasoner_query = {
        "message": {
            "query_graph": {
                "edges": {
                    "e00": {
                        "subject": "n00",
                        "object": "n01",
                        "predicates": ["biolink:physically_interacts_with"]
                    }
                },
                "nodes": {
                    "n00": {
                        "ids": ["CHEMBL.COMPOUND:CHEMBL112"]
                    },
                    "n01": {
                        "categories": ["biolink:Protein"]
                    }
                }
            }
        }
    }

    response = client.post('/query',
        data=json.dumps(reasoner_query),
        content_type='application/json')

    print(response.json)
    assert validate(response.json['message'], "Message", "1.1.0") == None
    assert len(response.json['message']['results']) == 0

# def test_post_embeddings():
#     """Test post embeddings to add embeddings to the model and rebuild it"""
    # curl -X POST "http://localhost:8808/embedding?types=Both&emb_name=test4&description=test&model_id=openpredict-baseline-omim-drugbank" -H  "accept: */*" -H  "Content-Type: multipart/form-data" -F "embedding_file=@neurodkg_embedding.json;type=application/json"
    # url = '/embedding?types=Both&emb_name=test_embedding&description=Embeddingdescription&model_id=openpredict-baseline-omim-drugbank'
    # files = {
    #     'embedding_file': ('neurodkg_embedding.json;type', open(embeddings_filepath + ';type', 'rb')),
    # }
    # headers = {
    #     'accept': '*/*',
    #     'Content-Type': 'multipart/form-data',
    # }
    # response = client.post(url, 
    #                         # files=('embedding_file', json.dumps(embeddings_json)), 
    #                         files=files,
    #                         headers=headers)
    #                         # content_type='application/json'))
    # print(response.status_code)
    # assert response.status_code == 200

