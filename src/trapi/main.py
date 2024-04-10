import logging
import os

from predict_drug_target import get_drug_target_predictions
from trapi_predict_kit import settings, TRAPI

# from drkg_model.api import api as drkg_model_api
from openpredict_model.api import api as openpredict_api
from openpredict_model.predict import get_predictions, get_similarities
from openpredict_model.utils import get_openpredict_dir

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

if not os.path.exists(get_openpredict_dir("input/drugbank-drug-goa.csv")):
    raise ValueError(
        "❌ The data required to run the prediction models could not be found in the `data` folder"
        "i️ Use `pip install dvc` and `dvc pull` to pull the data easily"
    )

log_level = logging.ERROR
# log_level = logging.INFO
logging.basicConfig(level=log_level)

openapi_info = {
    "contact": {
        "name": "Michel Dumontier",
        "email": "michel.dumontier@maastrichtuniversity.nl",
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
            # "inferred",
        ],
        "externalDocs": {
            "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
        },
    }
}

itrb_url_prefix = "openpredict"
app = TRAPI(
    predict_endpoints=[
        get_predictions,
        get_similarities,
        get_drug_target_predictions,
    ],
    info=openapi_info,
    itrb_url_prefix=itrb_url_prefix,
    opentelemetry=True,
    dev_server_url="https://openpredict.semanticscience.org",
    title='OpenPredict API',
    version='1.0.0',
    openapi_version='3.0.1',
    description="""Get predicted targets for a given entity: the **potential drugs treating a given disease**, or the **potential diseases a given drug could treat**.
\n\nUse the **predict** and **similarity** operations to easily retrieve predictions for a given entity (output format complying with the [BioThings Explorer](https://x-bte-extension.readthedocs.io/en/latest/x-bte-kgs-operations.html)).
\n\nPredictions are currently produced using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/) from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
\n\nMore documentation available at [github.com/MaastrichtU-IDS/translator-openpredict](https://github.com/MaastrichtU-IDS/translator-openpredict)
\n\n[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml)
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    trapi_description="""The default example TRAPI query will give you a list of predicted potential drug treatments for a given disease


You can also try this query to retrieve similar entities for a given drug:

```json
{
    "message": {
        "query_graph": {
            "edges": {
                "e01": {
                    "object": "n1",
                    "predicates": [ "biolink:similar_to" ],
                    "subject": "n0"
                }
            },
            "nodes": {
                "n0": {
                    "categories": [ "biolink:Drug" ],
                    "ids": [ "DRUGBANK:DB00394" ]
                },
                "n1": {
                    "categories": [ "biolink:Drug" ]
                }
            }
        }
    },
    "query_options": { "n_results": 5 }
}
```

Or this TRAPI query to get drug-target predictions:

```json
{
    "message": {
        "query_graph": {
            "edges": {"e01": {"object": "n1", "predicates": ["biolink:interacts_with"], "subject": "n0"}},
            "nodes": {
                "n0": {
                    "categories": ["biolink:Drug"],
                    "ids": ["PUBCHEM.COMPOUND:5329102", "PUBCHEM.COMPOUND:4039", "CHEMBL.COMPOUND:CHEMBL1431"]},
                "n1": {
                    "categories": ["biolink:Protein"],
                    "ids": ["UniProtKB:O75251"]
                }
            }
        }
    },
    "query_options": {"max_score": 1, "min_score": 0.1, "n_results": 10}
}
```
        """,
)

app.include_router(openpredict_api)
# app.include_router(drkg_model_api)
