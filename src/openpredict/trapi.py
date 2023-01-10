import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from reasoner_pydantic import Query

from openpredict.predict_output import PredictOptions
from openpredict.trapi_parser import resolve_trapi_query
from openpredict.utils import log


class TRAPI(FastAPI):
    """Translator Reasoner API - wrapper for FastAPI."""

    required_tags = [
        {"name": "reasoner"},
        {"name": "trapi"},
        {"name": "models"},
        {"name": "openpredict"},
        {"name": "translator"},
    ]


    def __init__(
        self,
        *args: Any,
        predict_endpoints: List[Callable],
        servers: Optional[List[Dict[str, str]]] = None,
        info: Optional[Dict[str, Any]] = None,
        title='Translator Reasoner API',
        version='1.0.0',
        openapi_version='3.0.1',
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
        self.servers = servers
        self.predict_endpoints = predict_endpoints
        self.info = info

        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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


        @self.post("/query", name="TRAPI query",
            description="""The default example TRAPI query will give you a list of predicted potential drug treatments for a given disease

You can also try this query to retrieve similar entities to a given drug:

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
        """,
            response_model=Query,
            tags=["reasoner"],
        )
        def post_reasoner_predict(
                request_body: Query = Body(..., example=TRAPI_EXAMPLE)
            ) -> Query:
            """Get predicted associations for a given ReasonerAPI query.

            :param request_body: The ReasonerStdAPI query in JSON
            :return: Predictions as a ReasonerStdAPI Message
            """
            query_graph = request_body.message.query_graph.dict(exclude_none=True)

            if len(query_graph["edges"]) == 0:
                return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
                # return ({"status": 400, "title": "Bad Request", "detail": "No edges", "type": "about:blank" }, 400)

            if len(query_graph["edges"]) > 1:
                # Currently just return a empty result if multi-edges query
                return {"message": {'knowledge_graph': {'nodes': {}, 'edges': {}}, 'query_graph': query_graph, 'results': []}}
                # return ({"status": 501, "title": "Not Implemented", "detail": "Multi-edges queries not yet implemented", "type": "about:blank" }, 501)

            reasonerapi_response = resolve_trapi_query(
                request_body.dict(exclude_none=True),
                self.predict_endpoints
            )

            return JSONResponse(reasonerapi_response) or ('Not found', 404)



        @self.get("/meta_knowledge_graph", name="Get the meta knowledge graph",
            description="Get the meta knowledge graph",
            response_model=dict,
            tags=["trapi"],
        )
        def get_meta_knowledge_graph() -> dict:
            """Get predicates and entities provided by the API

            :return: JSON with biolink entities
            """
            metakg = {
                'edges': [],
                'nodes': {}
            }
            log.info("IN TRAPI METAKG")
            log.info(self.predict_endpoints)
            print("IN TRAPI METAKG", predict_endpoints, flush=True)
            for predict_func in self.predict_endpoints:
                if predict_func._trapi_predict['edges'] not in metakg['edges']:
                    metakg['edges'] += predict_func._trapi_predict['edges']
                # Merge nodes dict
                metakg['nodes'] = {**metakg['nodes'], **predict_func._trapi_predict['nodes']}

            return JSONResponse(metakg)


        @self.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response


        @self.get("/health", include_in_schema=False)
        def health_check():
            """Health check for Translator elastic load balancer"""
            return {'status': 'ok'}


        @self.get("/", include_in_schema=False)
        def redirect_root_to_docs():
            """Redirect the route / to /docs"""
            return RedirectResponse(url='/docs')


        # Generate endpoints for the loaded models
        def endpoint_factory(predict_func):

            def prediction_endpoint(
                input_id: str = predict_func._trapi_predict['default_input'],
                model_id: str = predict_func._trapi_predict['default_model'],
                min_score: float = None, max_score: float = None,
                n_results: int = None
            ):
                try:
                    return predict_func(input_id, PredictOptions.parse_obj({
                        "model_id": model_id,
                        "min_score": min_score,
                        "max_score": max_score,
                        "n_results": n_results,
                        # "types": ['biolink:Drug'],
                    }))
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    return (f'Error when running the prediction: {e}', 500)

            return prediction_endpoint

        for predict_func in self.predict_endpoints:
            self.add_api_route(
                path=predict_func._trapi_predict['path'],
                methods=["GET"],
                # endpoint=copy_func(prediction_endpoint, model['path'].replace('/', '')),
                endpoint=endpoint_factory(predict_func),
                name=predict_func._trapi_predict['name'],
                openapi_extra={"description": predict_func._trapi_predict['description']},
                response_model=dict,
                tags=["models"],
            )



    def openapi(self) -> Dict[str, Any]:
        """Build custom OpenAPI schema."""
        if self.openapi_schema:
            return self.openapi_schema

        tags = self.required_tags
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
