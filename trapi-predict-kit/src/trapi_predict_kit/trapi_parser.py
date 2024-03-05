import re

from bmt import Toolkit

from trapi_predict_kit.config import settings
from trapi_predict_kit.utils import get_entities_labels, log

# TODO: add evidence path to TRAPI

biolink = Toolkit()


def get_biolink_parents(concept):
    concept_snakecase = concept.replace("biolink:", "")
    concept_snakecase = re.sub(r"(?<!^)(?=[A-Z])", "_", concept_snakecase).lower()
    try:
        return biolink.get_ancestors(
            name=concept_snakecase,
            reflexive=True,
            formatted=True,
            mixin=True,
        )
    except Exception as e:
        log.warn(f"Error getting parents of {concept_snakecase}, using the original IDs: {e}")
        return [concept]


def resolve_trapi_query(reasoner_query, endpoints_list, infores: str = ""):
    """Main function for TRAPI
    Convert an array of predictions objects to ReasonerAPI format
    Run the get_predict to get the QueryGraph edges and nodes
    {disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

    :param: reasoner_query Query from Reasoner API
    :return: Results as ReasonerAPI object
    """
    # Example TRAPI message: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/examples/Message/simple.json
    query_graph = reasoner_query["message"]["query_graph"]
    # Default query_options
    model_id = None
    n_results = None
    min_score = None
    max_score = None
    query_options = {}
    if "query_options" in reasoner_query:
        query_options = reasoner_query["query_options"]
        if "n_results" in query_options:
            n_results = int(query_options["n_results"])
        if "min_score" in query_options:
            min_score = float(query_options["min_score"])
        if "max_score" in query_options:
            max_score = float(query_options["max_score"])
        if "model_id" in query_options:
            model_id = str(query_options["model_id"])

    query_plan = {}
    # TODO: add a way to automatically resolve IDs passed to the prediction function?
    # resolved_ids_object = {}

    # Parse the query_graph to build the query plan
    for edge_id, qg_edge in query_graph.get("edges", {}).items():
        qg_subject_node_id = qg_edge.get("subject")
        qg_object_node_id = qg_edge.get("object")
        subject_node = query_graph["nodes"].get(qg_subject_node_id)
        object_node = query_graph["nodes"].get(qg_object_node_id)
        # resolved_ids_object = resolve_ids_with_nodenormalization_api(
        #     subject_node.get("ids", []) + object_node.get("ids", []), resolved_ids_object
        # )
        query_plan[edge_id] = {
            "subject": subject_node,
            "predicates": qg_edge.get("predicates"),
            "object": object_node,
            "qg_subject_node_id": qg_subject_node_id,
            "qg_object_node_id": qg_object_node_id,
        }

    knowledge_graph = {"nodes": {}, "edges": {}}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan:
        for predict_func in endpoints_list:
            # TODO: run the functions in parallel with future.concurrent?

            for func_edge in predict_func._trapi_predict["edges"]:
                predicate_parents = get_biolink_parents(func_edge["predicate"])
                subject_parents = get_biolink_parents(func_edge["subject"])
                object_parents = get_biolink_parents(func_edge["object"])

                subjs_to_predict = None
                pred_to_predict = None
                objs_to_predict = None
                log.debug(f"QUERY PLAN: {query_plan[edge_qg_id]}")
                # TODO: add support for "qualifier_constraints" on query edges. cf. https://github.com/NCATSTranslator/testing/blob/main/ars-requests/not-none/1.2/mvp2cMetformin.json
                if (
                    any(i in predicate_parents for i in query_plan[edge_qg_id]["predicates"])
                    and any(i in subject_parents for i in query_plan[edge_qg_id]["subject"].get("categories", []))
                    and any(i in object_parents for i in query_plan[edge_qg_id]["object"].get("categories", []))
                ):
                    subjs_to_predict = query_plan[edge_id]["subject"]
                    pred_to_predict = func_edge["predicate"]
                    objs_to_predict = query_plan[edge_id]["object"]

                inverse = False
                if "inverse" in func_edge:
                    inverse_parents = get_biolink_parents(func_edge["inverse"])
                    if (
                        any(i in inverse_parents for i in query_plan[edge_qg_id]["predicates"])
                        and any(i in object_parents for i in query_plan[edge_qg_id]["subject"].get("categories", []))
                        and any(i in subject_parents for i in query_plan[edge_qg_id]["object"].get("categories", []))
                    ):
                        inverse = True
                        subjs_to_predict = query_plan[edge_id]["object"]
                        pred_to_predict = func_edge["inverse"]
                        objs_to_predict = query_plan[edge_id]["subject"]
                        # Also inverse the node binding IDs
                        # qg_subject_node_id, qg_object_node_id = qg_object_node_id, qg_subject_node_id
                        query_plan[edge_id]["qg_subject_node_id"], query_plan[edge_id]["qg_object_node_id"] = (
                            query_plan[edge_id]["qg_object_node_id"],
                            query_plan[edge_id]["qg_subject_node_id"],
                        )

                # Check if requested subject/predicate/object are served by the function
                if subjs_to_predict and pred_to_predict and objs_to_predict:
                    subject_ids = subjs_to_predict.get("ids", [])
                    object_ids = objs_to_predict.get("ids", [])

                    try:
                        log.info(f"🔮⏳️ Getting predictions for: {subject_ids} | {object_ids}")
                        # Run function to get predictions
                        prediction_results = predict_func(
                            {
                                "subjects": subject_ids,
                                "objects": object_ids,
                                "options": {
                                    "model_id": model_id,
                                    "min_score": min_score,
                                    "max_score": max_score,
                                    "n_results": n_results,
                                    # "subject_types": subjs_to_predict.get("categories", []),
                                    # "object_types": objs_to_predict.get("categories", []),
                                },
                            }
                        )
                        prediction_json = prediction_results["hits"]
                    except Exception as e:
                        log.error(f"Error getting the predictions: {e}")
                        prediction_json = []

                    # Get the labels of all entities returned by the prediction function
                    all_ids = [pred["subject"] for pred in prediction_json] + [
                        pred["object"] for pred in prediction_json
                    ]
                    labels_dict = get_entities_labels(list(set(all_ids)))

                    for association in prediction_json:
                        # id/type of nodes are registered in a dict to avoid duplicate in knowledge_graph.nodes
                        subject_id = association["subject"]
                        object_id = association["object"]

                        # TODO: XAI get path between source and target nodes (first create the function for this)

                        # If the target ID is given, we filter here from the predictions
                        # if 'to_kg_id' in query_plan[edge_qg_id] and target_node_id not in query_plan[edge_qg_id]['to_kg_id']:
                        if (
                            "subject_kg_id" in query_plan[edge_id]
                            and "object_kg_id" in query_plan[edge_id]
                            and object_id not in query_plan[edge_qg_id]["object_kg_id"]
                        ):
                            pass

                        else:
                            edge_kg_id = "e" + str(kg_edge_count)
                            # Get the ID of the predicted entity in result association
                            # based on the type expected for the association "to" node

                            node_dict[subject_id] = {
                                "type": association.get(
                                    "subject_type", subjs_to_predict.get("categories", ["biolink:NamedThing"])
                                ),
                            }
                            node_dict[object_id] = {
                                "type": association.get(
                                    "object_type", objs_to_predict.get("categories", ["biolink:NamedThing"])
                                ),
                            }

                            if "subject_label" in association:
                                node_dict[subject_id]["label"] = association["subject_label"]
                            else:
                                if subject_id in labels_dict and labels_dict[subject_id]:
                                    node_dict[subject_id]["label"] = labels_dict[subject_id]["id"]["label"]

                            if "object_label" in association:
                                node_dict[object_id]["label"] = association["object_label"]
                            else:
                                if object_id in labels_dict and labels_dict[object_id]:
                                    node_dict[object_id]["label"] = labels_dict[object_id]["id"]["label"]

                            # edge_association_type = 'biolink:ChemicalToDiseaseOrPhenotypicFeatureAssociation'
                            # relation = 'RO:0002434' # interacts with
                            # relation = 'OBOREL:0002606'
                            association_score = float(association["score"])

                            model_id_label = model_id
                            if not model_id_label:
                                model_id_label = "openpredict_baseline"

                            edge_dict = {}
                            # Map the source/target of query_graph to source/target of association
                            # if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                            if inverse:
                                edge_dict["subject"] = object_id
                                edge_dict["object"] = subject_id
                            else:
                                edge_dict["subject"] = subject_id
                                edge_dict["object"] = object_id

                            edge_dict["predicate"] = pred_to_predict

                            # See attributes examples: https://github.com/NCATSTranslator/Evidence-Provenance-Confidence-Working-Group/blob/master/attribute_epc_examples/COHD_TRAPI1.1_Attribute_Example_2-3-21.yml
                            edge_dict = {
                                **edge_dict,
                                # TODO: not required anymore? 'association_type': edge_association_type,
                                # 'relation': relation,
                                # More details on attributes: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#attribute-
                                "sources": [
                                    {
                                        "resource_id": infores,
                                        "resource_role": "primary_knowledge_source",
                                    },
                                    {"resource_id": "infores:cohd", "resource_role": "supporting_data_source"},
                                ],
                                "attributes": [
                                    {
                                        "description": "model_id",
                                        "attribute_type_id": "EDAM:data_1048",
                                        "value": model_id_label,
                                    },
                                    {
                                        # TODO: use has_confidence_level?
                                        "description": "score",
                                        "attribute_type_id": "EDAM:data_1772",
                                        "value": association_score
                                        # https://www.ebi.ac.uk/ols/ontologies/edam/terms?iri=http%3A%2F%2Fedamontology.org%2Fdata_1772&viewMode=All&siblings=false
                                    },
                                    # https://github.com/NCATSTranslator/ReasonerAPI/blob/1.4/ImplementationGuidance/Specifications/knowledge_level_agent_type_specification.md
                                    {
                                        "attribute_type_id": "biolink:agent_type",
                                        "value": "computational_model",
                                        "attribute_source": infores,
                                    },
                                    {
                                        "attribute_type_id": "biolink:knowledge_level",
                                        "value": "prediction",
                                        "attribute_source": infores,
                                    },
                                ],
                                # "knowledge_types": knowledge_types
                            }

                            # Add the association in the knowledge_graph as edge
                            # Use the type as key in the result association dict (for IDs)
                            knowledge_graph["edges"][edge_kg_id] = edge_dict

                            # Add the bindings to the results object
                            result = {
                                "node_bindings": {},
                                "analyses": [
                                    {
                                        # TODO: pass infores_curie
                                        "resource_id": infores,
                                        "score": association_score,
                                        # "dummy_score": 0.42,
                                        "scoring_method": "Model confidence between 0 and 1",
                                        "edge_bindings": {edge_qg_id: [{"id": edge_kg_id}]},
                                    }
                                ],
                            }
                            result["node_bindings"][query_plan[edge_id]["qg_subject_node_id"]] = [
                                {"id": association["subject"]}
                            ]
                            result["node_bindings"][query_plan[edge_id]["qg_object_node_id"]] = [
                                {"id": association["object"]}
                            ]
                            query_results.append(result)

                            kg_edge_count += 1
                            if kg_edge_count == n_results:
                                break

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_category = properties["type"]
        if isinstance(node_category, str) and not node_category.startswith("biolink:"):
            node_category = "biolink:" + node_category.capitalize()
        if isinstance(node_category, str):
            node_category = [node_category]
        node_to_add = {
            "categories": node_category,
        }
        if "label" in properties and properties["label"]:
            node_to_add["name"] = properties["label"]
        knowledge_graph["nodes"][node_id] = node_to_add

    return {
        "message": {"knowledge_graph": knowledge_graph, "query_graph": query_graph, "results": query_results},
        "query_options": query_options,
        "reasoner_id": infores,
        "schema_version": settings.TRAPI_VERSION,
        "biolink_version": settings.BIOLINK_VERSION,
        "status": "Success",
        # "logs": [
        #     {
        #         "code": None,
        #         "level": "INFO",
        #         "message": "No descendants found from Ontology KP for QNode 'n00'.",
        #         "timestamp": "2023-04-05T07:24:26.646711"
        #     },
        # ]
    }
