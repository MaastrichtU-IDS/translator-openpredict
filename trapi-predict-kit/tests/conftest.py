import logging
import os

from trapi_predict_kit import TRAPI, PredictInput, PredictOutput, trapi_predict
from trapi_predict_kit.config import settings

# Setup logger
log_level = logging.INFO
logging.basicConfig(level=log_level)
os.environ["VIRTUAL_HOST"] = "openpredict.semanticscience.org"


# Define additional metadata to integrate this function in TRAPI
@trapi_predict(
    path="/predict",
    name="Get predicted targets for a given entity",
    description="Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.",
    # Define which edges can be predicted by this function in a TRAPI query
    edges=[
        {
            "subject": "biolink:Drug",
            "predicate": "biolink:treats",
            "inverse": "biolink:treated_by",
            "object": "biolink:Disease",
        },
    ],
    nodes={"biolink:Disease": {"id_prefixes": ["OMIM"]}, "biolink:Drug": {"id_prefixes": ["DRUGBANK"]}},
)
def get_predictions(request: PredictInput) -> PredictOutput:
    # Predictions results should be a list of entities
    # for which there is a predicted association with the input entity
    predictions = {
        "hits": [
            {
                "subject": "drugbank:DB00001",
                "object": "OMIM:246300",
                "subject_type": "biolink:Drug",
                "score": 0.12345,
                "label": "Leipirudin",
            }
        ],
        "count": 1,
    }
    return predictions


openapi_info = {
    "contact": {
        "name": "{{cookiecutter.author_name}}",
        "email": "{{cookiecutter.author_email}}",
        # "x-id": "{{cookiecutter.author_orcid}}",
        "x-role": "responsible developer",
    },
    "license": {
        "name": "MIT license",
        "url": "https://opensource.org/licenses/MIT",
    },
    "termsOfService": "https://github.com/{{cookiecutter.github_organization_name}}/{{cookiecutter.package_name}}/blob/main/LICENSE.txt",
    "x-translator": {
        "component": "KP",
        # TODO: update the Translator team to yours
        "team": ["Clinical Data Provider"],
        "biolink-version": settings.BIOLINK_VERSION,
        "infores": "infores:openpredict",
        "externalDocs": {
            "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
        },
    },
    "x-trapi": {
        "version": settings.TRAPI_VERSION,
        "asyncquery": False,
        "operations": [
            "lookup",
        ],
        "externalDocs": {
            "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
        },
    },
}

app = TRAPI(
    predict_endpoints=[get_predictions],
    info=openapi_info,
    title="TRAPI predict kit dev",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""TRAPI predict kit development
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    itrb_url_prefix="openpredict",
    dev_server_url="https://openpredict.semanticscience.org",
    # opentelemetry=True,
    # docker run -d --name jaeger -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one:latest
)
