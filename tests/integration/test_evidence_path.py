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


def test_get_evidence_path():
    """Test predict API GET operation for a drug"""

    drug_id = "DRUGBANK:DB00915"
    disease_id = "OMIM:104300"
    url = f'/evidence_path?drug_id={drug_id}&disease_id={disease_id}'
    response = client.get(url).json()
    assert len(response['hits']) >= 1

