import os
import logging
from datetime import datetime
from openpredict.openpredict_utils import init_openpredict_dir
from openpredict.rdf_utils import init_triplestore, retrieve_features, retrieve_models
from openpredict.openpredict_model import addEmbedding, get_predictions, get_similarities, load_similarity_embedding_models
from openpredict.reasonerapi_parser import typed_results_to_reasonerapi

from fastapi import FastAPI, Body, Request, Response, Query
from fastapi.openapi.utils import get_openapi
# from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from reasoner_pydantic import Query, Message


TRAPI_EXAMPLE = {
  "message": {
    "query_graph": {
      "edges": {
        "e01": {
          "object": "n1",
          "predicates": [
            "biolink:treated_by"
          ],
          "subject": "n0"
        }
      },
      "nodes": {
        "n0": {
          "categories": [
            "biolink:Disease"
          ],
          "ids": [
            "OMIM:246300",
            "MONDO:0007190"
          ]
        },
        "n1": {
          "categories": [
            "biolink:Drug"
          ]
        }
      }
    }
  },
  "query_options": {
    "max_score": 1,
    "min_score": 0.5,
    "n_results": 10
  }
}



# def custom_openapi(app):
#     if app.openapi_schema:
#         return app.openapi_schema
    
#     openapi_schema = get_openapi(
#         title='OpenPredict API',
#         version='1.0.0',
#         openapi_version='3.0.1',
#         description="""Get predicted targets for a given entity: the **potential drugs treating a given disease**, or the **potential diseases a given drug could treat**.
#         \n\n* Use the `/predict` operation to easily retrieve predictions for a given entity (operation annotated for the [BioThings Explorer](https://x-bte-extension.readthedocs.io/en/latest/x-bte-kgs-operations.html)).
#         \n\n* Predictions are currently produced using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/) from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
#         \n\n* You can also install the [OpenPredict Python package](https://pypi.org/project/openpredict/) to train and serve a new model yourself.
#         \n\n* More documentation available at [github.com/MaastrichtU-IDS/translator-openpredict](https://github.com/MaastrichtU-IDS/translator-openpredict)
#         \n\n[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml)
#         \n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
#         routes=app.routes,
#         tags=[
#             {"name": "translator"},
#             {"name": "trapi"},
#             {"name": "reasoner"},
#         ],
#     )
#     openapi_schema["servers"] = [
#         {
#             "url": 'https://openpredict.semanticscience.org',
#             "description": 'Production OpenPredict TRAPI',
#             "x-maturity": 'production',
#             "x-location": 'IDS'
#         }
#     ]

#     openapi_schema["info"]["x-translator"] = {
#         "component": 'KP',
#         "team": "Clinical Data Provider",
#         "externalDocs": {
#             "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
#             "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
#         },
#         "infores": 'infores:openpredict',
#     }
#     openapi_schema["info"]["x-trapi"] = {
#         "version": "1.2.0",
#         "externalDocs": {
#             "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
#             "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
#         },
#         "operations": [
#             "lookup",
#         ],
#     }
#     # if self.trapi_operations:
#     #     openapi_schema["info"]["x-trapi"]["operations"] = self.trapi_operations
#     openapi_schema["info"]["contact"] = {
#         "name": "Vincent Emonet",
#         "email": "vincent.emonet@maastrichtuniversity.nl",
#         # "x-id": "vemonet",
#         "x-role": "responsible developer",
#     }
#     openapi_schema["info"]["termsOfService"] = 'https://raw.githubusercontent.com/MaastrichtU-IDS/translator-openpredict/master/LICENSE'


#     # From fastapi:
#     openapi_schema["info"]["x-logo"] = {
#         "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
#     }

#     app.openapi_schema = openapi_schema
#     return app.openapi_schema
