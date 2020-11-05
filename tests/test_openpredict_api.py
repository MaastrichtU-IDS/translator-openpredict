import pytest
import connexion
import json
import pathlib
from requests_toolbelt import MultipartEncoder
from openpredict.openpredict_utils import init_openpredict_dir
from openpredict.openpredict_model import addEmbedding

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

def test_post_embeddings():
    """Test add embeddings to the model and rebuild it"""
    embeddings_filepath = str(pathlib.Path(__file__).parent.joinpath("data/neurodkg_embedding.json"))
    
    with open(embeddings_filepath,  encoding="utf8") as embeddings_file:
        run_id = addEmbedding(embeddings_file, 'test_embedding', 'Both', 'test embedding', 'openpredict-baseline-omim-drugbank')
        assert len(run_id) == 36
    
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

