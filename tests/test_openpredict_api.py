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
    url = '/predict?entity=DRUGBANK:DB00394&model_id=openpredict-baseline-omim-drugbank&n_results=42'
    response = client.get(url)
    assert len(response.json['results']) == 42
    assert response.json['count'] == 42
    assert response.json['results'][0]['target']['id'] == 'OMIM:246300'

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
            # "has_confidence_level": 0.5
            # }
        }
    }
    response = client.post(url, 
                           data=json.dumps(reasoner_query), 
                           content_type='application/json')
    edges = response.json['knowledge_graph']['edges']
    assert len(edges) == 300
    assert edges[0]['target_id'] == 'OMIM:246300'

def test_post_embeddings(client):
    """Test POST embedding call to add embeddings to the model and rebuild it"""
    url = '/embedding?types=Both&emb_name=test_embedding&description=Embedding%20description&model_id=openpredict-baseline-omim-drugbank'
    with open('tests/data/neurodkg_embedding.json',  encoding="utf8") as data_file:
        json_data = json.load(data_file)
        response = client.post(url, 
                            data=json_data, 
                            content_type='application/json')
        print(response.status_code)
        assert response.status_code == 200