import logging
import os

from trapi_predict_kit import TRAPI, settings

from predict_drug_target.predict import get_drug_target_predictions
from predict_drug_target.train import train
from predict_drug_target.utils import COLLECTIONS
from predict_drug_target.vectordb import init_vectordb

log_level = logging.INFO
logging.basicConfig(level=log_level)

# TODO: remove, not used, predict imported in trapi-openpredict

trapi_example = {
    "message": {
        "query_graph": {
            "edges": {"e01": {"object": "n1", "predicates": ["biolink:interacts_with"], "subject": "n0"}},
            "nodes": {
                "n0": {
                    "categories": ["biolink:Drug"],
                    "ids": ["PUBCHEM.COMPOUND:5329102", "PUBCHEM.COMPOUND:4039", "CHEMBL.COMPOUND:CHEMBL1431"]
                },
                "n1": {
                    "categories": ["biolink:Protein"],
                    "ids": ["UniProtKB:O75251"]
                }
            }
        }
    },
    "query_options": {"max_score": 1, "min_score": 0.1, "n_results": 10}
}


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
    "termsOfService": "https://github.com/MaastrichtU-IDS/predict-drug-target/blob/main/LICENSE.txt",
    "x-translator": {
        "component": "KP",
        "team": ["Clinical Data Provider"],
        "biolink-version": settings.BIOLINK_VERSION,
        "infores": "infores:predict-drug-target",
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
    predict_endpoints=[get_drug_target_predictions],
    info=openapi_info,
    title="Predict Drug Target interactions TRAPI",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""Get predicted protein targets for a given drug
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    itrb_url_prefix="predict-drug-target",
    dev_server_url="https://predict-drug-target.137.120.31.160.nip.io",
    trapi_example=trapi_example,
    # trapi_description=""
)


@app.post("/reset-vectordb", name="Reset vector database", description="Reset the collections in the vectordb")
def post_reset_vectordb(api_key: str):
    init_vectordb(recreate=True, api_key=api_key)
    return {"status": "ok"}


@app.post("/train", name="Run training", description="Run training of the model")
def post_train(api_key: str):
    # init_vectordb(recreate=True, api_key=api_key)
    scores = train()
    # return scores_df.to_dict(orient="records")

    return scores




# def configure_otel(app):
#     # open telemetry https://github.com/ranking-agent/aragorn/blob/main/src/otel_config.py#L4
#     # https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/deployment-guide/monitoring/
#     # https://github.com/TranslatorSRI/Jaeger-demo
#     if not os.environ.get("NO_JAEGER"):
#         logging.info("Starting up jaeger telemetry")
#         import warnings
#         from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
#         from opentelemetry import trace
#         from opentelemetry.exporter.jaeger.thrift import JaegerExporter
#         from opentelemetry.sdk.resources import SERVICE_NAME as telemetery_service_name_key, Resource
#         from opentelemetry.sdk.trace import TracerProvider
#         from opentelemetry.sdk.trace.export import BatchSpanProcessor
#         from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

#         service_name = os.environ.get("OTEL_SERVICE_NAME", "OPENPREDICT")
#         # httpx connections need to be open a little longer by the otel decorators
#         # but some libs display warnings of resource being unclosed.
#         # these supresses such warnings.
#         logging.captureWarnings(capture=True)
#         warnings.filterwarnings("ignore",category=ResourceWarning)
#         trace.set_tracer_provider(
#             TracerProvider(
#                 resource=Resource.create({telemetery_service_name_key: service_name})
#             )
#         )
#         jaeger_host = os.environ.get('JAEGER_HOST', 'jaeger-otel-agent.sri')
#         jaeger_port = int(os.environ.get('JAEGER_PORT', '6831'))
#         jaeger_exporter = JaegerExporter(
#             agent_host_name=jaeger_host,
#             agent_port=jaeger_port,
#         )
#         trace.get_tracer_provider().add_span_processor(
#             BatchSpanProcessor(jaeger_exporter)
#         )
#         # tracer = trace.get_tracer(__name__)
#         FastAPIInstrumentor.instrument_app(app, tracer_provider=trace, excluded_urls="docs,openapi.json")
#         HTTPXClientInstrumentor().instrument()

# # Configure open telemetry if enabled
# configure_otel(app)
