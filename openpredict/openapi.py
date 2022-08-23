from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from reasoner_pydantic import Message, Query

from openpredict.config import settings
from openpredict.openpredict_model import (
    load_similarity_embeddings,
    load_treatment_classifier,
    load_treatment_embeddings,
)

# from rdflib_endpoint import SparqlEndpoint

# from openpredict.utils import init_openpredict_dir
# from openpredict.rdf_utils import init_triplestore
# import logging
# from datetime import datetime


# class TRAPI(SparqlEndpoint):
class TRAPI(FastAPI):
    """Translator Reasoner API - wrapper for FastAPI."""

    # Embeddings and classifier are loaded here at the start of the API 
    baseline_model_treatment: str
    treatment_embeddings = None
    treatment_classifier = None

    baseline_model_similarity: str
    similarity_embeddings = None

    required_tags = [
        {"name": "reasoner"},
        {"name": "trapi"},
        {"name": "biothings"},
        {"name": "openpredict"},
        {"name": "translator"},
    ]

    def __init__(
        self,
        *args,
        baseline_model_treatment: Optional[str] = 'openpredict-baseline-omim-drugbank',
        baseline_model_similarity: Optional[str] = 'openpredict-baseline-omim-drugbank',
        # contact: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            title='OpenPredict API',
            root_path_in_servers=False,
            **kwargs,
        )
        self.baseline_model_treatment = baseline_model_treatment
        self.baseline_model_similarity = baseline_model_similarity
        # Initialize embeddings features and classifiers to be used by the API
        self.treatment_embeddings = load_treatment_embeddings(baseline_model_treatment)
        self.treatment_classifier = load_treatment_classifier(baseline_model_treatment)
        self.similarity_embeddings = load_similarity_embeddings()
        
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def openapi(self) -> Dict[str, Any]:
        """Build custom OpenAPI schema."""
        if self.openapi_schema:
            return self.openapi_schema

        tags = self.required_tags
        if self.openapi_tags:
            tags += self.openapi_tags
    
        openapi_schema = get_openapi(
            title='OpenPredict API',
            version='1.0.0',
            openapi_version='3.0.1',
            description="""Get predicted targets for a given entity: the **potential drugs treating a given disease**, or the **potential diseases a given drug could treat**.
            \n\nUse the **predict** and **similarity** operations to easily retrieve predictions for a given entity (output format complying with the [BioThings Explorer](https://x-bte-extension.readthedocs.io/en/latest/x-bte-kgs-operations.html)).
            \n\nPredictions are currently produced using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/) from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
            \n\nMore documentation available at [github.com/MaastrichtU-IDS/translator-openpredict](https://github.com/MaastrichtU-IDS/translator-openpredict)
            \n\n[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml)
            \n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
            routes=self.routes,
            tags=tags,
        )

        if not settings.DEV_MODE:
          openapi_schema["servers"] = [
              {
                  "url": settings.PROD_URL,
                  "description": 'OpenPredict TRAPI ITRB Production Server',
                  "x-maturity": 'production'
              },
              {
                  "url": settings.TEST_URL,
                  "description": 'OpenPredict TRAPI ITRB Test Server',
                  "x-maturity": 'testing'
              },
              {
                  "url": settings.STAGING_URL,
                  "description": 'OpenPredict TRAPI ITRB CI Server',
                  "x-maturity": 'staging'
              },
              {
                  "url": settings.DEV_URL,
                  "description": 'OpenPredict TRAPI ITRB Development Server',
                  "x-maturity": 'development',
                  "x-location": 'IDS'
              },
          ]

        openapi_schema["info"]["x-translator"] = {
            "component": 'KP',
            "team": ["Clinical Data Provider"],
            "biolink-version": settings.BIOLINK_VERSION,
            "infores": 'infores:openpredict',
            "externalDocs": {
                "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
                "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
            },
        }
        openapi_schema["info"]["x-trapi"] = {
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

        openapi_schema["info"]["contact"] = {
            "name": "Vincent Emonet",
            "email": "vincent.emonet@maastrichtuniversity.nl",
            # "x-id": "vemonet",
            "x-role": "responsible developer",
        }
        openapi_schema["info"]["termsOfService"] = 'https://raw.githubusercontent.com/MaastrichtU-IDS/translator-openpredict/master/LICENSE'
        openapi_schema["info"]["license"] = {
            "name": "MIT license",
            "url": "https://opensource.org/licenses/MIT",
        }

        # To make the /predict call compatible with the BioThings Explorer:
        openapi_schema["paths"]["/predict"]["x-bte-kgs-operations"] = [
          {
            "inputs": [
              {
                "id": "biolink:DRUGBANK",
                "semantic": "biolink:ChemicalSubstance"
              }
            ],
            "outputs": [
              {
                "id": "biolink:OMIM",
                "semantic": "biolink:Disease"
              }
            ],
            "parameters": {
              "drug_id": "{inputs[0]}"
            },
            "predicate": "biolink:treats",
            "supportBatch": False,
            "responseMapping": {
              "OMIM": "hits.id"
            }
          },
          {
            "inputs": [
              {
                "id": "biolink:OMIM",
                "semantic": "biolink:Disease"
              }
            ],
            "outputs": [
              {
                "id": "biolink:DRUGBANK",
                "semantic": "biolink:ChemicalSubstance"
              }
            ],
            "parameters": {
              "disease_id": "{inputs[0]}"
            },
            "predicate": "biolink:treated_by",
            "supportBatch": False,
            "responseMapping": {
              "DRUGBANK": "hits.id"
            }
          }
        ]
        
        # From fastapi:
        openapi_schema["info"]["x-logo"] = {
            # "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
            "url": "https://raw.githubusercontent.com/MaastrichtU-IDS/dsri-documentation/master/website/static/img/um_logo.png"
        }

        self.openapi_schema = openapi_schema
        return self.openapi_schema


class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


class SimilarityTypes(str, Enum):
    Drugs = "Drugs"
    Diseases = "Diseases"

class FeatureTypesDrugs(str,Enum):
    PPI_SIM = "PPI-SIM"
    TC = "TC"
    SE_SIM = "SE-SIM"
    TARGETSEQ_SIM = "TARGETSEQ-SIM"
    GO_SIM = "GO-SIM"

class FeatureTypesDiseases(str, Enum) : 
    HPO_SIM = "HPO-SIM"
    PHENO_SIM = "PHENO-SIM"


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
            # "MONDO:0007190"
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

