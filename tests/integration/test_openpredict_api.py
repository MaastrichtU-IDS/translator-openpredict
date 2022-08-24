import json
import os

import pkg_resources
import pytest
from fastapi.testclient import TestClient
from openpredict.config import settings
from openpredict.main import app
from openpredict.rdf_utils import init_triplestore
from openpredict.utils import init_openpredict_dir
from reasoner_validator import validate

# Create and start Flask from openapi.yml before running tests
init_openpredict_dir()
# init_triplestore()

client = TestClient(app)


def test_get_predict_drug():
    """Test predict API GET operation for a drug"""
    url = '/predict?drug_id=DRUGBANK:DB00394&model_id=openpredict-baseline-omim-drugbank&n_results=42'
    response = client.get(url).json()
    assert len(response['hits']) == 42
    assert response['count'] == 42
    assert response['hits'][0]['type'] == 'disease'


def test_get_predict_disease():
    """Test predict API GET operation for a disease"""
    url = '/predict?disease_id=OMIM:246300&model_id=openpredict-baseline-omim-drugbank&n_results=42'
    response = client.get(url).json()
    assert len(response['hits']) == 42
    assert response['count'] == 42
    assert response['hits'][0]['type'] == 'drug'


# docker-compose exec api pytest tests/integration/test_openpredict_api.py::test_get_similarity_drug -s
def test_get_similarity_drug():
    """Test prediction similarity API GET operation for a drug"""
    n_results=5
    url = '/similarity?drug_id=DRUGBANK:DB00394&types=Drugs&model_id=drugs_fp_embed.txt&n_results=' + str(n_results)
    response = client.get(url).json()
    assert len(response['hits']) == n_results
    assert response['count'] == n_results
    assert response['hits'][0]['type'] == 'drug'

# docker-compose exec api pytest tests/integration/test_openpredict_api.py::test_get_similarity_disease -s
def test_get_similarity_disease():
    """Test prediction similarity API GET operation for a disease"""
    n_results=5
    url = '/similarity?disease_id=OMIM:246300&types=Diseases&model_id=disease_hp_embed.txt&n_results=' + str(n_results)
    response = client.get(url).json()
    assert len(response['hits']) == n_results
    assert response['count'] == n_results
    assert response['hits'][0]['type'] == 'disease'


def test_post_trapi():
    """Test Translator ReasonerAPI query POST operation to get predictions"""
    print(f'Testing for TRAPI version {settings.TRAPI_VERSION_TEST} 🏷️')
    url = '/query'
    for trapi_filename in os.listdir(pkg_resources.resource_filename('tests', 'queries')):
        with open(pkg_resources.resource_filename('tests', 'queries/' + trapi_filename),'r') as f:
            reasoner_query = f.read()
            response = client.post(
                url, 
                data=reasoner_query, 
                headers={"Content-Type": "application/json"},
                # content_type='application/json'
            )

            # print(response.json)
            edges = response.json()['message']['knowledge_graph']['edges'].items()
            # print(response)
            print(trapi_filename)
            assert validate(response.json()['message'], "Message", settings.TRAPI_VERSION_TEST) == None
            if trapi_filename.endswith('limit3.json'):
                assert len(edges) == 3
            elif trapi_filename.endswith('limit1.json'):
                assert len(edges) == 1
            else:
                assert len(edges) >= 5


def test_trapi_empty_response():
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
        headers={"Content-Type": "application/json"},
        # content_type='application/json'
    )

    print(response.json)
    assert validate(response.json()['message'], "Message", settings.TRAPI_VERSION_TEST) == None
    assert len(response.json()['message']['results']) == 0

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

