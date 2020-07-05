import pytest
import connexion

# Create and start Flask OpenAPI before running tests
flask_app = connexion.FlaskApp(__name__)
flask_app.add_api('../openapi.yml')
@pytest.fixture(scope='module')
def client():
    with flask_app.app.test_client() as c:
        yield c

def test_get_predict_drug_disease(client):
    """Test prediction call for drug-disease"""
    url = "/v1/predict/drug-disease?disease=test_disease&drug=testdrug1"
    expected_json = {"drug": "testdrug1", "disease": "test_disease", "score": 0.8}
    response = client.get(url)

    assert response.json == expected_json