import logging

from trapi_predict_kit import TRAPI, PredictInput, PredictOutput, trapi_predict
from trapi_predict_kit.config import settings

log_level = logging.INFO
logging.basicConfig(level=log_level)


@trapi_predict(
    path="/predict",
    name="Get predicted targets for a given entity",
    description="Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.",
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
    predictions = []
    # Add the code the load the model and get predictions here
    # Available props: request.subjects, request.objects, request.options
    for subj in request.subjects:
        predictions.append(
            {
                "subject": subj,
                "object": "OMIM:246300",
                "score": 0.12345,
            }
        )
    for obj in request.objects:
        predictions.append(
            {
                "subject": "DRUGBANK:DB00001",
                "object": obj,
                "score": 0.12345,
            }
        )
    return {"hits": predictions, "count": len(predictions)}


openapi_info = {
    "contact": {
        "name": "Firstname Lastname",
        "email": "email@example.com",
        # "x-id": "https://orcid.org/0000-0000-0000-0000",
        "x-role": "responsible developer",
    },
    "license": {
        "name": "MIT license",
        "url": "https://opensource.org/licenses/MIT",
    },
    "termsOfService": "https://github.com/your-org-or-username/my-model/blob/main/LICENSE.txt",
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
    title="OpenPredict TRAPI",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""Machine learning models to produce predictions that can be integrated to Translator Reasoner APIs.
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    itrb_url_prefix="openpredict",
    dev_server_url="https://openpredict.semanticscience.org",
)
