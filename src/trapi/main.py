import logging
import os
from enum import Enum

from fastapi import File, UploadFile
from rdflib import Graph

from drkg_model.api import api as drkg_model_api
from openpredict.config import settings
from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict.trapi import TRAPI
from openpredict.utils import init_openpredict_dir
from openpredict_model.api import api as openpredict_api
from openpredict_model.predict import get_predictions, get_similarities
from openpredict_model.train import add_embedding


class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

init_openpredict_dir()

log_level = logging.ERROR
if settings.DEV_MODE:
    log_level = logging.INFO
logging.basicConfig(level=log_level)

predict_endpoints = [
    get_predictions,
    get_similarities,
]

models_g = Graph()
models_g.parse("models/openpredict_baseline.ttl")


openapi_info = {
    "contact": {
        "name": "Vincent Emonet",
        "email": "vincent.emonet@maastrichtuniversity.nl",
        # "x-id": "vemonet",
        "x-role": "responsible developer",
    },
    "license": {
        "name": "MIT license",
        "url": "https://opensource.org/licenses/MIT",
    },
    "termsOfService": 'https://raw.githubusercontent.com/MaastrichtU-IDS/translator-openpredict/master/LICENSE',

    "x-translator": {
        "component": 'KP',
        "team": ["Clinical Data Provider"],
        "biolink-version": settings.BIOLINK_VERSION,
        "infores": 'infores:openpredict',
        "externalDocs": {
            "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
        },
    },
    "x-trapi": {
        "version": settings.TRAPI_VERSION,
        "asyncquery": False,
        # TODO: cf. https://github.com/NCATSTranslator/ReasonerAPI/pull/339
        # "test_data_location": "",
        "operations": [
            "lookup",
        ],
        "externalDocs": {
            "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
        },
    }
}



app = TRAPI(
    predict_endpoints=predict_endpoints,
    info=openapi_info,
    title='OpenPredict API',
    version='1.0.0',
    openapi_version='3.0.1',
    description="""Get predicted targets for a given entity: the **potential drugs treating a given disease**, or the **potential diseases a given drug could treat**.
\n\nUse the **predict** and **similarity** operations to easily retrieve predictions for a given entity (output format complying with the [BioThings Explorer](https://x-bte-extension.readthedocs.io/en/latest/x-bte-kgs-operations.html)).
\n\nPredictions are currently produced using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/) from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
\n\nMore documentation available at [github.com/MaastrichtU-IDS/translator-openpredict](https://github.com/MaastrichtU-IDS/translator-openpredict)
\n\n[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml)
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
)

app.include_router(openpredict_api)
app.include_router(drkg_model_api)



@app.get("/features", name="Return the features trained in the models",
    description="""Return the features trained in the model, for Drugs, Diseases or Both.""",
    response_model=dict,
    tags=["openpredict"],
)
def get_features(embedding_type: EmbeddingTypes ='Drugs') -> dict:
    """Get features in the model

    :return: JSON with features
    """
    if type(embedding_type) is EmbeddingTypes:
        embedding_type = embedding_type.value
    return retrieve_features(models_g, embedding_type)



@app.get("/models", name="Return the models with their training features and scores",
    description="""Return the models with their training features and scores""",
    response_model=dict,
    tags=["openpredict"],
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models(models_g)



@app.post("/embedding", name="Upload your embedding for drugs or diseases",
    description="""Upload your embedding file:

1. Select which types do you have in the embeddings: Drugs, Diseases or Both.

2. Define the base `model_id`: use the `/models` call to see the list of trained models with their characteristics, and pick the ID of the model you will use as base to add your embedding

3. The model will be retrained and evaluation will be stored in a triplestore (available in `/models`)
""",
    response_model=dict,
    tags=["openpredict"],
)
def post_embedding(
        emb_name: str, description: str,
        types: EmbeddingTypes ='Both', model_id: str ='openpredict_baseline',
        apikey: str=None,
        uploaded_file: UploadFile = File(...)
    ) -> dict:
    """Post JSON embeddings via the API, with simple APIKEY authentication
    provided in environment variables
    """
    if type(types) is EmbeddingTypes:
        types = types.value

    # Ignore the API key check if no env variable defined (for development)
    if os.getenv('OPENPREDICT_APIKEY') == apikey or os.getenv('OPENPREDICT_APIKEY') is None:
        embedding_file = uploaded_file.file
        run_id, scores = add_embedding(
            embedding_file, emb_name, types, model_id)
        print('Embeddings uploaded')
        # train_model(False)
        return {
            'status': 200,
            'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(scores)
        }
    else:
        return {'Forbidden': 403}
