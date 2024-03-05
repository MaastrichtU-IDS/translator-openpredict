from fastapi.testclient import TestClient

from trapi_predict_kit import settings

from .conftest import app

client = TestClient(app)

validator = None
try:
    from reasoner_validator.validator import TRAPIResponseValidator

    # NOTE: Validate only prod because validate requires py3.9+ and OpenPredict requires 3.8
    validator = TRAPIResponseValidator(
        trapi_version=settings.TRAPI_VERSION,
        # If None, then the current Biolink Model Toolkit default release applies
        biolink_version=settings.BIOLINK_VERSION,
        # 'sources' are set to trigger checking of expected edge knowledge source provenance
        # sources={
        #     "ara_source": "infores:molepro",
        #     "kp_source": "infores:knowledge-collaboratory",
        #     "kp_source_type": "primary"
        # },
        # None let the system decide the default validation strictness by validation context
        strict_validation=None,
    )
except Exception:
    print("reasoner-validator not found, not running TRAPI response validation checks")


def check_trapi_compliance(response):
    if validator:
        # validator.check_compliance_of_trapi_response(response["message"])
        validator.check_compliance_of_trapi_response(response)
        validator_resp = validator.get_messages()
        print("⚠️ REASONER VALIDATOR WARNINGS:")
        print(validator_resp["warnings"])
        if len(validator_resp["errors"]) == 0:
            print("✅ NO REASONER VALIDATOR ERRORS")
        else:
            print("🧨 REASONER VALIDATOR ERRORS")
            print(validator_resp["errors"])
        assert len(validator_resp["errors"]) == 0


def test_get_predict_drug():
    """Test predict API GET operation for a drug"""
    response = client.post(
        "/predict",
        json={
            "subjects": ["DRUGBANK:DB00394"],
            "options": {
                "model_id": "openpredict_baseline",
            },
        },
    ).json()
    assert len(response["hits"]) >= 1
    assert response["count"] >= 1
    assert response["hits"][0]["subject"] == "drugbank:DB00001"


def test_get_meta_kg():
    """Get the metakg"""
    response = client.get("/meta_knowledge_graph").json()
    assert len(response["edges"]) >= 1
    assert len(response["nodes"]) >= 1


def test_post_trapi():
    """Test Translator ReasonerAPI query POST operation to get predictions"""
    trapi_query = {
        "message": {
            "query_graph": {
                "edges": {"e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:treats"]}},
                "nodes": {
                    "n0": {"categories": ["biolink:Drug"]},
                    "n1": {"ids": ["OMIM:246300"], "categories": ["biolink:Disease"]},
                },
            }
        },
        "query_options": {
            "n_results": 3,
            "min_score": 0,
            "max_score": 1,
        },
    }
    response = client.post(
        "/query",
        json=trapi_query,
        headers={"Content-Type": "application/json"},
    ).json()
    edges = response["message"]["knowledge_graph"]["edges"].items()
    assert len(edges) >= 1
    check_trapi_compliance(response)


def test_trapi_empty_response():
    trapi_query = {
        "message": {
            "query_graph": {
                "edges": {
                    "e00": {"subject": "n00", "object": "n01", "predicates": ["biolink:physically_interacts_with"]}
                },
                "nodes": {"n00": {"ids": ["CHEMBL.COMPOUND:CHEMBL112"]}, "n01": {"categories": ["biolink:Protein"]}},
            }
        }
    }
    response = client.post(
        "/query",
        json=trapi_query,
        headers={"Content-Type": "application/json"},
    )
    assert len(response.json()["message"]["results"]) == 0


def test_healthcheck():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_root_redirect():
    response = client.get("/")
    assert response.status_code == 200
