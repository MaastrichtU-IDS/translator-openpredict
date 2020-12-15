import pytest
import connexion
import json
from openpredict.openpredict_utils import init_openpredict_dir

# Create and start Flask from openapi.yml before running tests
init_openpredict_dir()
flask_app = connexion.FlaskApp(__name__)
flask_app.add_api('../openpredict/openapi.yml')
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

def test_post_reasoner_predict(client):
    """Test ReasonerAPI query POST operation to get predictions"""
    url = '/query'
    reasoner_query = {
        "message": {
            # "n_results": 10,
            "query_graph": {
                "edges": [
                    {
                    "id": "e00",
                    "source_id": "n00",
                    "target_id": "n01",
                    "type": "treated_by"
                    }
                ],
                "nodes": [
                    {
                    "curie": "DRUGBANK:DB00394",
                    "id": "n00",
                    "type": "drug"
                    },
                    {
                    "id": "n01",
                    "type": "disease"
                    }
                ]
            }
            # "query_options": {
            # "min_score": 0.5
            # }
        }
    }
    response = client.post(url, 
                           data=json.dumps(reasoner_query), 
                           content_type='application/json')
    edges = response.json['knowledge_graph']['edges']
    assert len(edges) == 300
    assert edges[0]['target_id'] == 'OMIM:246300'

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

