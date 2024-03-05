import os
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from reasoner_pydantic import Query

from trapi_predict_kit.trapi_parser import resolve_trapi_query
from trapi_predict_kit.types import PredictInput
from trapi_predict_kit.config import settings

REQUIRED_TAGS = [
    {"name": "reasoner"},
    {"name": "trapi"},
    {"name": "translator"},
]

default_trapi_example = {
    "message": {
        "query_graph": {
            "edges": {"e01": {"object": "n1", "predicates": ["biolink:treated_by"], "subject": "n0"}},
            "nodes": {
                "n0": {
                    "categories": ["biolink:Disease"],
                    "ids": [
                        "OMIM:246300",
                        # "MONDO:0007190"
                    ],
                },
                "n1": {"categories": ["biolink:Drug"]},
            },
        }
    },
    "query_options": {"max_score": 1, "min_score": 0.5, "n_results": 10},
}
default_trapi_description = (
    "Query the prediction endpoint with [TRAPI queries](https://github.com/NCATSTranslator/ReasonerAPI)"
)


class TRAPI(FastAPI):
    """Translator Reasoner API - wrapper for FastAPI."""

    def __init__(
        self,
        *args: Any,
        predict_endpoints: List[Callable],
        ordered_servers: Optional[List[Dict[str, str]]] = None,
        itrb_url_prefix: Optional[str] = None,
        dev_server_url: Optional[str] = None,
        trapi_example: Optional[Query] = None,
        trapi_description: str = default_trapi_description,
        info: Optional[Dict[str, Any]] = None,
        title="Translator Reasoner API",
        version="1.0.0",
        openapi_version="3.0.1",
        opentelemetry=False,
        description="""Get predicted targets for a given entity
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            title=title,
            version=version,
            openapi_version=openapi_version,
            description=description,
            root_path_in_servers=False,
            **kwargs,
        )
        if itrb_url_prefix and opentelemetry:
            add_opentelemetry(self, itrb_url_prefix)

        self.predict_endpoints = predict_endpoints
        self.info = info
        self.infores = self.info.get("x-translator", {}).get("infores")
        if not self.infores and itrb_url_prefix:
            self.infores = f"infores:{itrb_url_prefix}"
        self.openapi_version = openapi_version
        self.trapi_example = trapi_example if trapi_example else default_trapi_example
        self.trapi_description = trapi_description

        # On ITRB deployment and local dev we directly use the current server
        self.servers = []

        # For the API deployed on our server and registered to SmartAPI we provide the complete list
        if os.getenv("VIRTUAL_HOST"):
            if itrb_url_prefix:
                self.servers.append(
                    {
                        "url": f"https://{itrb_url_prefix}.transltr.io",
                        "description": "TRAPI ITRB Production Server",
                        "x-maturity": "production",
                    }
                )
                self.servers.append(
                    {
                        "url": f"https://{itrb_url_prefix}.test.transltr.io",
                        "description": "TRAPI ITRB Test Server",
                        "x-maturity": "testing",
                    }
                )
                self.servers.append(
                    {
                        "url": f"https://{itrb_url_prefix}.ci.transltr.io",
                        "description": "TRAPI ITRB CI Server",
                        "x-maturity": "staging",
                    }
                )
            if dev_server_url:
                self.servers.append(
                    {"url": dev_server_url, "description": "TRAPI Dev Server", "x-maturity": "development"}
                )

            ordered_servers = []
            # Add the current server as 1st server in the list
            for server in self.servers:
                if os.getenv("VIRTUAL_HOST") in server["url"]:
                    ordered_servers.append(server)
                    break
            # Add other servers
            for server in self.servers:
                if os.getenv("VIRTUAL_HOST") not in server["url"]:
                    ordered_servers.append(server)
            self.servers = ordered_servers

        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.post(
            "/query",
            name="TRAPI query",
            description=self.trapi_description,
            response_model=Query,
            tags=["reasoner"],
        )
        def post_reasoner_predict(request_body: Query = Body(..., example=self.trapi_example)) -> Query:
            """Get predicted associations for a given ReasonerAPI query.

            :param request_body: The ReasonerStdAPI query in JSON
            :return: Predictions as a ReasonerStdAPI Message
            """
            query_graph = request_body.message.query_graph.dict(exclude_none=True)

            if len(query_graph["edges"]) == 0:
                return {
                    "message": {
                        "knowledge_graph": {"nodes": {}, "edges": {}},
                        "query_graph": query_graph,
                        "results": [],
                    }
                }
                # return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)

            if len(query_graph["edges"]) > 1:
                # Currently just return a empty result if multi-edges query
                return {
                    "message": {
                        "knowledge_graph": {"nodes": {}, "edges": {}},
                        "query_graph": query_graph,
                        "results": [],
                    }
                }
                # return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)

            reasonerapi_response = resolve_trapi_query(
                request_body.dict(exclude_none=True), self.predict_endpoints, self.infores
            )

            return JSONResponse(reasonerapi_response) or ("Not found", 404)

        @self.get(
            "/meta_knowledge_graph",
            name="Get the meta knowledge graph",
            description="Get the meta knowledge graph",
            response_model=dict,
            tags=["trapi"],
        )
        def get_meta_knowledge_graph() -> dict:
            """Get predicates and entities provided by the API

            :return: JSON with biolink entities
            """
            metakg = {"edges": [], "nodes": {}}
            for predict_func in self.predict_endpoints:
                for func_edge in predict_func._trapi_predict["edges"]:
                    meta_edge = [
                        {
                            "subject": func_edge.get("subject"),
                            "predicate": func_edge.get("predicate"),
                            "object": func_edge.get("object"),
                        }
                    ]
                    if "inverse" in predict_func._trapi_predict and predict_func._trapi_predict["inverse"]:
                        meta_edge.append(
                            {
                                "subject": func_edge.get("object"),
                                "predicate": func_edge.get("inverse"),
                                "object": func_edge.get("subject"),
                            }
                        )

                if meta_edge not in metakg["edges"]:
                    metakg["edges"] += meta_edge
                # Merge nodes dict
                metakg["nodes"] = {**metakg["nodes"], **predict_func._trapi_predict["nodes"]}
            return JSONResponse(metakg)

        @self.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["Server-Timing"] = f"total;dur={process_time}"
            return response

        @self.get("/health", include_in_schema=False)
        def health_check():
            """Health check for Translator elastic load balancer"""
            return {"status": "ok"}

        @self.get("/", include_in_schema=False)
        def redirect_root_to_docs():
            """Redirect the route / to /docs"""
            return RedirectResponse(url="/docs")

        # Generate endpoints for the loaded models
        def endpoint_factory(predict_func):
            def prediction_endpoint(request: PredictInput):
                try:
                    return predict_func(PredictInput.parse_obj(request))
                except Exception as e:
                    return (f"Error when getting the predictions: {e}", 500)

            return prediction_endpoint

        for predict_func in self.predict_endpoints:
            self.add_api_route(
                path=predict_func._trapi_predict["path"],
                methods=["POST"],
                endpoint=endpoint_factory(predict_func),
                name=predict_func._trapi_predict["name"],
                openapi_extra={"description": predict_func._trapi_predict["description"]},
                response_model=dict,
                tags=["models"],
            )

    def openapi(self) -> Dict[str, Any]:
        """Build custom OpenAPI schema."""
        if self.openapi_schema:
            return self.openapi_schema

        tags = REQUIRED_TAGS
        if self.openapi_tags:
            tags += self.openapi_tags

        openapi_schema = get_openapi(
            # **self.info,
            title=self.title,
            version=self.version,
            openapi_version=self.openapi_version,
            description=self.description,
            routes=self.routes,
            tags=tags,
        )

        openapi_schema["servers"] = self.servers
        openapi_schema["info"] = {**openapi_schema["info"], **self.info}
        self.openapi_schema = openapi_schema
        return self.openapi_schema


def add_opentelemetry(app: FastAPI, service_name: str) -> None:
    """Configure Open Telemetry
    https://github.com/ranking-agent/aragorn/blob/main/src/otel_config.py#L4
    https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/deployment-guide/monitoring/
    https://github.com/TranslatorSRI/Jaeger-demo"""
    import logging
    import warnings

    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    logging.info("Starting up jaeger telemetry")
    # service_name = os.environ.get("OTEL_SERVICE_NAME", service_name)
    # httpx connections need to be open a little longer by the otel decorators
    # but some libs display warnings of resource being unclosed.
    # these supresses such warnings.
    logging.captureWarnings(capture=True)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    trace.set_tracer_provider(TracerProvider(resource=Resource.create({SERVICE_NAME: service_name})))
    jaeger_host = os.environ.get("JAEGER_HOST", "jaeger-otel-agent.sri")
    # jaeger_host = os.environ.get("JAEGER_HOST", "localhost")
    jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.get_tracer(__name__)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="docs,openapi.json")
    # FastAPIInstrumentor.instrument_app(app, tracer_provider=trace, excluded_urls="docs,openapi.json")
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
