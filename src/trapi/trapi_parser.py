import re

import requests

from openpredict.config import settings
from openpredict.utils import get_entities_labels
from trapi.loaded_models import models_list

# TODO: add evidence path to TRAPI

def is_accepted_id(id_to_check):
    if id_to_check.lower().startswith('omim') or id_to_check.lower().startswith('drugbank'):
        return True
    else:
        return False


def get_biolink_parents(concept):
    concept_snakecase = concept.replace('biolink:', '')
    concept_snakecase = re.sub(r'(?<!^)(?=[A-Z])', '_', concept_snakecase).lower()
    query_url = f'https://bl-lookup-sri.renci.org/bl/{concept_snakecase}/ancestors'
    try:
        resolve_curies = requests.get(query_url,
                            params={'version': f'v{settings.BIOLINK_VERSION}'})
        resp = resolve_curies.json()
        resp.append(concept)
        return resp
    except Exception:
        print(f'Error querying {query_url}, using the original IDs')
        return [concept]


def resolve_ids_with_nodenormalization_api(resolve_ids_list, resolved_ids_object):
    resolved_ids_list = []
    ids_to_normalize = []
    for id_to_resolve in resolve_ids_list:
        if is_accepted_id(id_to_resolve):
            resolved_ids_list.append(id_to_resolve)
            resolved_ids_object[id_to_resolve] = id_to_resolve
        else:
            ids_to_normalize.append(id_to_resolve)

    # Query Translator NodeNormalization API to convert IDs to OMIM/DrugBank IDs
    if len(ids_to_normalize) > 0:
        try:
            resolve_curies = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                                params={'curie': ids_to_normalize})
            # Get corresponding OMIM IDs for MONDO IDs if match
            resp = resolve_curies.json()
            for resolved_id, alt_ids in resp.items():
                for alt_id in alt_ids['equivalent_identifiers']:
                    if is_accepted_id(str(alt_id['identifier'])):
                        resolved_ids_list.append(str(alt_id['identifier']))
                        resolved_ids_object[str(alt_id['identifier'])] = resolved_id
        except Exception:
            print('Error querying the NodeNormalization API, using the original IDs')

    return resolved_ids_list, resolved_ids_object


def resolve_id(id_to_resolve, resolved_ids_object):
    if id_to_resolve in resolved_ids_object.keys():
        return resolved_ids_object[id_to_resolve]
    return id_to_resolve


def resolve_trapi_query(reasoner_query):
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
    if 'query_options' in reasoner_query.keys():
        if 'n_results' in reasoner_query["query_options"]:
            n_results = int(reasoner_query["query_options"]["n_results"])
        if 'min_score' in reasoner_query["query_options"]:
            min_score = float(reasoner_query["query_options"]["min_score"])
        if 'max_score' in reasoner_query["query_options"]:
            max_score = float(reasoner_query["query_options"]["max_score"])
        if 'model_id' in reasoner_query["query_options"]:
            model_id = int(reasoner_query["query_options"]["model_id"])

    query_plan = {}
    resolved_ids_object = {}

    # if not similarity_embeddings or similarity_embeddings == {}:
    # similarity_embeddings = None
    # treatment_embeddings = None

    # Parse the query_graph to build the query plan
    for edge_id, qg_edge in query_graph["edges"].items():
        # Build dict with all infos of associations to predict
        query_plan[edge_id] = {
            # 'predicates': qg_edge['predicates'],
            # 'qedge_subjects': qg_edge['subject'],
            'qg_source_id': qg_edge['subject'],
            'qg_target_id': qg_edge['object']
        }
        if 'predicates' in qg_edge.keys():
            query_plan[edge_id]['predicates'] = qg_edge['predicates']
        else:
            # Quick fix: in case no relation is provided
            query_plan[edge_id]['predicates'] = ['biolink:treats']

        # If single value provided for predicate: make it an array
        # if not isinstance(query_plan[edge_id]['predicate'], list):
        #     query_plan[edge_id]['predicate'] = [ query_plan[edge_id]['predicate'] ]

        # Get the nodes infos in the query plan object
        for node_id, node in query_graph["nodes"].items():
        # for node in query_graph['nodes']:
            if node_id == qg_edge['subject'] or node_id == qg_edge['object']:
                # if node_id == qg_edge['subject']:
                if 'ids' in node and 'from_qg_id' not in query_plan[edge_id].keys():
                    # TOREMOVE: If single values provided for id or category: make it an array
                    # if not isinstance(node['id'], list):
                    #     node['id'] = [ node['id'] ]

                    # Resolve the curie provided with the NodeNormalization API
                    query_plan[edge_id]['from_kg_id'], resolved_ids_object = resolve_ids_with_nodenormalization_api(node['ids'], resolved_ids_object)

                    query_plan[edge_id]['from_qg_id'] = node_id
                    if 'categories' in node.keys():
                        query_plan[edge_id]['from_type'] = node['categories']
                    else:
                        query_plan[edge_id]['from_type'] = 'biolink:NamedThing'
                    # TOREMOVE: handling of single values
                    # if not isinstance(query_plan[edge_id]['from_type'], list):
                    #     query_plan[edge_id]['from_type'] = [ query_plan[edge_id]['from_type'] ]

                elif 'to_qg_id' not in query_plan[edge_id].keys():
                    # The node without curie is the association's "to"
                    query_plan[edge_id]['to_qg_id'] = node_id

                    if 'ids' in node.keys():
                        query_plan[edge_id]['to_kg_id'], resolved_ids_object = resolve_ids_with_nodenormalization_api(node['ids'], resolved_ids_object)

                    if 'categories' in node.keys():
                        query_plan[edge_id]['to_type'] = node['categories']
                    else:
                        query_plan[edge_id]['to_type'] = ['biolink:NamedThing']
                    if not isinstance(query_plan[edge_id]['to_type'], list):
                        query_plan[edge_id]['to_type'] = [ query_plan[edge_id]['to_type'] ]

    knowledge_graph = {'nodes': {}, 'edges': {}}
    node_dict = {}
    query_results = []
    kg_edge_count = 0
    # supportedCategories = ['biolink:Drug', 'biolink:Disease', 'biolink:NamedThing', 'biolink:ChemicalSubstance']

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan.keys():

        for (do_prediction, model_metadata) in models_list:
            for relation in model_metadata['relations']:
                predicate_parents = get_biolink_parents(relation['predicate'])

                if any(i in predicate_parents for i in query_plan[edge_qg_id]['predicates']):
                    # Resolve when asking for treats prediction
                    types_parents = []
                    for type in query_plan[edge_qg_id]['from_type'] + query_plan[edge_qg_id]['to_type']:
                        types_parents += get_biolink_parents(type)
                    # types_parents = get_biolink_parents(query_plan[edge_qg_id]['from_type']) + get_biolink_parents(query_plan[edge_qg_id]['to_type'])
                    if any(i in types_parents for i in query_plan[edge_qg_id]['from_type']) and any(i in types_parents for i in query_plan[edge_qg_id]['to_type']):
                        # Iterate over the list of ids provided
                        for id_to_predict in query_plan[edge_qg_id]['from_kg_id']:
                            labels_dict = get_entities_labels([id_to_predict])
                            label_to_predict = None
                            if id_to_predict in labels_dict:
                                label_to_predict = labels_dict[id_to_predict]['id']['label']
                            try:
                                # Run OpenPredict to get predictions
                                # prediction_json = do_prediction(
                                #     id_to_predict, model_id,
                                #     min_score, max_score, n_results=None
                                # )
                                # TODO: add options param
                                prediction_results = do_prediction(
                                    id_to_predict,
                                    {
                                        "model_id": model_id,
                                        "min_score": min_score,
                                        "max_score": max_score,
                                        "n_results": n_results,
                                        "types": query_plan[edge_qg_id]['from_type'],
                                    }
                                )
                                prediction_json = prediction_results['hits']
                            except Exception as e:
                                print(f"Error getting the predictions: {e}")
                                import traceback
                                print(traceback.format_exc())
                                prediction_json = []

                            for association in prediction_json:
                                # id/type of nodes are registered in a dict to avoid duplicate in knowledge_graph.nodes
                                # Build dict of node ID : label
                                source_node_id = resolve_id(id_to_predict, resolved_ids_object)
                                target_node_id = resolve_id(association['id'], resolved_ids_object)

                                # TODO: XAI get path between source and target nodes (first create the function for this)

                                # If the target ID is given, we filter here from the predictions
                                if 'to_kg_id' in query_plan[edge_qg_id] and target_node_id not in query_plan[edge_qg_id]['to_kg_id']:
                                    pass

                                else:
                                    edge_kg_id = 'e' + str(kg_edge_count)
                                    # Get the ID of the predicted entity in result association
                                    # based on the type expected for the association "to" node
                                    # node_dict[id_to_predict] = query_plan[edge_qg_id]['from_type']
                                    # node_dict[association[query_plan[edge_qg_id]['to_type']]] = query_plan[edge_qg_id]['to_type']
                                    node_dict[source_node_id] = {
                                        'type': query_plan[edge_qg_id]['from_type']
                                    }
                                    if label_to_predict:
                                        node_dict[source_node_id]['label'] = label_to_predict

                                    node_dict[target_node_id] = {
                                        'type': association['type']
                                    }
                                    if 'label' in association.keys():
                                        node_dict[target_node_id]['label'] = association['label']
                                    else:
                                        # TODO: improve to avoid to call the resolver everytime
                                        labels_dict = get_entities_labels([target_node_id])
                                        if target_node_id in labels_dict.keys() and labels_dict[target_node_id]:
                                            node_dict[target_node_id]['label'] = labels_dict[target_node_id]['id']['label']

                                    # edge_association_type = 'biolink:ChemicalToDiseaseOrPhenotypicFeatureAssociation'
                                    relation = 'RO:0002434' # interacts with
                                    # relation = 'OBOREL:0002606'
                                    association_score = str(association['score'])

                                    model_id_label = model_id
                                    if not model_id_label:
                                        model_id_label = "openpredict-baseline-omim-drugbank"

                                    # See attributes examples: https://github.com/NCATSTranslator/Evidence-Provenance-Confidence-Working-Group/blob/master/attribute_epc_examples/COHD_TRAPI1.1_Attribute_Example_2-3-21.yml
                                    edge_dict = {
                                        # TODO: not required anymore? 'association_type': edge_association_type,
                                        # 'relation': relation,

                                        # More details on attributes: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#attribute-
                                        'attributes': [
                                            {
                                                "description": "model_id",
                                                "attribute_type_id": "EDAM:data_1048",
                                                "value": model_id_label
                                            },
                                            {
                                                # TODO: use has_confidence_level?
                                                "description": "score",
                                                "attribute_type_id": "EDAM:data_1772",
                                                "value": association_score
                                                # https://www.ebi.ac.uk/ols/ontologies/edam/terms?iri=http%3A%2F%2Fedamontology.org%2Fdata_1772&viewMode=All&siblings=false
                                            },
                                            {
                                                'attribute_type_id': 'biolink:aggregator_knowledge_source',
                                                'value': 'infores:openpredict',
                                                'value_type_id': 'biolink:InformationResource',
                                                'attribute_source': 'infores:openpredict',
                                                # 'value_url': 'https://openpredict.semanticscience.org/query'
                                            },
                                            {
                                                'attribute_type_id': 'biolink:supporting_data_source',
                                                'value': 'infores:cohd',
                                                'value_type_id': 'biolink:InformationResource',
                                                'attribute_source': 'infores:openpredict',
                                                # 'value_url': 'https://openpredict.semanticscience.org'
                                            },
                                        ]
                                    }

                                    # Map the source/target of query_graph to source/target of association
                                    # if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                                    edge_dict['subject'] = source_node_id
                                    edge_dict['object'] = target_node_id

                                    # TODO: Define the predicate depending on the association source type returned by OpenPredict classifier
                                    if len(query_plan[edge_qg_id]['predicates']) > 0:
                                        edge_dict['predicate'] = query_plan[edge_qg_id]['predicates'][0]
                                    else:
                                        edge_dict['predicate'] = relations['predicate']

                                    # if query_plan[edge_qg_id]['from_type'] == 'drug':
                                    #     # and 'biolink:Drug' in query_plan[edge_qg_id]['predicates']: ?
                                    #     edge_dict['predicate'] = query_plan[edge_qg_id]['predicate']
                                    # else:
                                    #     edge_dict['predicate'] = 'biolink:treated_by'

                                    # Add the association in the knowledge_graph as edge
                                    # Use the type as key in the result association dict (for IDs)
                                    knowledge_graph['edges'][edge_kg_id] = edge_dict

                                    # Add the bindings to the results object
                                    result = {'edge_bindings': {}, 'node_bindings': {}}
                                    result['edge_bindings'][edge_qg_id] = [
                                        {
                                            "id": edge_kg_id
                                        }
                                    ]
                                    result['node_bindings'][query_plan[edge_qg_id]['from_qg_id']] = [
                                        {
                                            "id": source_node_id
                                        }
                                    ]
                                    result['node_bindings'][query_plan[edge_qg_id]['to_qg_id']] = [
                                        {
                                            "id": target_node_id
                                        }
                                    ]
                                    query_results.append(result)

                                    kg_edge_count += 1
                                    if kg_edge_count == n_results:
                                        break
                    else:
                        print('BioLink category not parents of Drug or Disease, no results returned')
                        prediction_json = []

        else:
            prediction_json = []

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_category = properties['type']
        if isinstance(node_category, str) and not node_category.startswith('biolink:'):
            node_category = 'biolink:' + node_category.capitalize()
        if isinstance(node_category, str):
            node_category = [ node_category ]
        node_to_add = {
            'categories': node_category ,
        }
        if 'label' in properties and properties['label']:
            node_to_add['name'] = properties['label']
        knowledge_graph['nodes'][node_id] = node_to_add

    return {"message": {'knowledge_graph': knowledge_graph, 'query_graph': query_graph, 'results': query_results}}


example_trapi = {
  "message": {
    "query_graph": {
      "edges": {
        "e01": {
          "object": "n1",
          "predicates": ["biolink:treated_by", "biolink:treats"],
          "subject": "n0"
        }
      },
      "nodes": {
        "n0": {
          "categories": ["biolink:Disease", "biolink:Drug"],
          "ids": ["OMIM:246300", "DRUGBANK:DB00394"]
        },
        "n1": {
          "categories": ["biolink:Drug", "biolink:Disease"]
        }
      }
    }
  },
  "query_options": {
    "max_score": 1,
    "min_score": 0.5
  }
}
