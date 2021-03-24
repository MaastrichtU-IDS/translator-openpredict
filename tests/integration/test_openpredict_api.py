import pytest
import pkg_resources
import connexion
import json
from openpredict.openpredict_utils import init_openpredict_dir

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
    tests_list = [
        {'limit': 3, 'class': 'drug'},
        {'limit': 'no', 'class': 'drug'},
        {'limit': 3, 'class': 'disease'},
        {'limit': 'no', 'class': 'disease'},
    ]
    for trapi_test in tests_list:
        trapi_filename = 'trapi_' + trapi_test['class'] + '_limit' + str(trapi_test['limit']) + '.json'
        with open(pkg_resources.resource_filename('tests', 'queries/' + trapi_filename),'r') as f:
            reasoner_query = f.read()
            response = client.post(url, 
                                    data=reasoner_query, 
                                    content_type='application/json')

            print(response.json)
            edges = response.json['message']['knowledge_graph']['edges'].items()
            if trapi_test['limit'] == 'no':
                assert len(edges) >= 300
            else:
                assert len(edges) == trapi_test['limit']


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

