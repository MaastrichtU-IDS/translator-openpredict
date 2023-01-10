import logging

from drkg_model.api import api as drkg_model_api
from openpredict.config import settings
from openpredict.trapi import TRAPI
from openpredict.utils import init_openpredict_dir
from openpredict_model.api import api as openpredict_api
from openpredict_model.predict import get_predictions, get_similarities

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

servers_list = [
    {
        "url": settings.PROD_URL,
        "description": 'TRAPI ITRB Production Server',
        "x-maturity": 'production'
    },
    {
        "url": settings.TEST_URL,
        "description": 'TRAPI ITRB Test Server',
        "x-maturity": 'testing'
    },
    {
        "url": settings.STAGING_URL,
        "description": 'TRAPI ITRB CI Server',
        "x-maturity": 'staging'
    },
    {
        "url": settings.DEV_URL,
        "description": 'TRAPI ITRB Development Server',
        "x-maturity": 'development',
        # "x-location": 'IDS'
    },
]

# Order the servers list based on the env variable used for nginx proxy in docker
if settings.VIRTUAL_HOST:
    servers = []
    # Add the current server as 1st server in the list
    for server in servers_list:
        if settings.VIRTUAL_HOST in server['url']:
            servers.append(server)
            break
    # Add other servers
    for server in servers_list:
        if settings.VIRTUAL_HOST not in server['url']:
            servers.append(server)
else:
    servers = []


app = TRAPI(
    predict_endpoints=predict_endpoints,
    info=openapi_info,
    servers=servers,
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
