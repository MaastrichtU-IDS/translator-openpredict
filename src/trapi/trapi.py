from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from reasoner_pydantic import Query

from trapi.trapi_parser import resolve_trapi_query

app = APIRouter()


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


@app.post("/query", name="TRAPI query",
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

    reasonerapi_response = resolve_trapi_query(request_body.dict(exclude_none=True))

    return JSONResponse(reasonerapi_response) or ('Not found', 404)



@app.get("/meta_knowledge_graph", name="Get the meta knowledge graph",
    description="Get the meta knowledge graph",
    response_model=dict,
    tags=["trapi"],
)
def get_meta_knowledge_graph() -> dict:
    """Get predicates and entities provided by the API

    :return: JSON with biolink entities
    """
    openpredict_predicates = {
        "edges": [
            {
                "object": "biolink:Disease",
                "predicate": "biolink:treats",
                "relations": [
                    "RO:0002434"
                ],
                "subject": "biolink:Drug"
                # TODO: https://github.com/NCATSTranslator/ReasonerAPI/pull/331/files
                # "knowledge_types": ['inferred', 'lookup']
            },
            {
                "object": "biolink:Drug",
                "predicate": "biolink:treated_by",
                "relations": [
                    "RO:0002434"
                ],
                "subject": "biolink:Disease"
            },
            {
                "object": "biolink:Entity",
                "predicate": "biolink:similar_to",
                # "relations": [
                #     "RO:0002434"
                # ],
                "subject": "biolink:Entity"
            },
        ],
        "nodes": {
            "biolink:Disease": {
                "id_prefixes": [
                    "OMIM"
                ]
            },
            "biolink:Drug": {
                "id_prefixes": [
                    "DRUGBANK"
                ]
            }
        }
    }

    return JSONResponse(openpredict_predicates)
