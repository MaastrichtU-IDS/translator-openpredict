import pytest
import connexion

# Create and start Flask OpenAPI before running tests
flask_app = connexion.FlaskApp(__name__)
flask_app.add_api('../openapi.yml')
@pytest.fixture(scope='module')
def client():
    with flask_app.app.test_client() as c:
        yield c

def test_get_predict(client):
    """Test predict API call"""
    url = "/v1/predict?entity=DB00394&input_type=drug&predict_type=disease"
    # expected_json = {
    #     'results': [{'source' : 'test_disease', 'target': 'associated drug 1', 'score': 0.8}],
    #     'count': 1
    # }
    response = client.get(url)

    assert len(response.json) > 1
    # assert response.json == expected_json