from openpredict.openpredict_model import get_predictions
import requests

def is_accepted_id(id_to_check):
    if id_to_check.lower().startswith('omim') or id_to_check.lower().startswith('drugbank'):
        return True
    else:
        return False

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
        resolve_curies = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                            params={'curie': ids_to_normalize})
        # Get corresponding OMIM IDs for MONDO IDs if match
        resp = resolve_curies.json()
        for resolved_id, alt_ids in resp.items():
            for alt_id in alt_ids['equivalent_identifiers']:
                if is_accepted_id(str(alt_id['identifier'])):
                    resolved_ids_list.append(str(alt_id['identifier']))
                    resolved_ids_object[str(alt_id['identifier'])] = resolved_id
                    # print('Mapped ' + resolved_id + ' - "' + alt_ids['id']['label'] + '" to ' + alt_id['identifier'])
    return resolved_ids_list, resolved_ids_object

def resolve_id(id_to_resolve, resolved_ids_object):
    if id_to_resolve in resolved_ids_object.keys():
        return resolved_ids_object[id_to_resolve]
    return id_to_resolve

def typed_results_to_reasonerapi(reasoner_query):
    """Convert an array of predictions objects to ReasonerAPI format
    Run the get_predict to get the QueryGraph edges and nodes
    {disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

    :param: reasoner_query Query from Reasoner API
    :return: Results as ReasonerAPI object
    """
    # Example TRAPI message: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/examples/Message/simple.json
    query_graph = reasoner_query["message"]["query_graph"]
    # Default query_options
    model_id = 'openpredict-baseline-omim-drugbank'
    n_results = None
    min_score = None
    max_score = None
    if 'query_options' in reasoner_query.keys():
        query_options = reasoner_query["query_options"]
        if 'n_results' in reasoner_query["query_options"]:
            n_results = int(reasoner_query["query_options"]["n_results"])
        if 'min_score' in reasoner_query["query_options"]:
            min_score = float(reasoner_query["query_options"]["min_score"])
        if 'max_score' in reasoner_query["query_options"]:
            max_score = float(reasoner_query["query_options"]["max_score"])

    query_plan = {}
    resolved_ids_object = {}

    # Parse the query_graph to build the query plan
    for edge_id, qg_edge in query_graph["edges"].items():
        # Build dict with all infos of associations to predict 
        query_plan[edge_id] = {
            'predicate': qg_edge['predicate'],
            # 'qedge_subjects': qg_edge['subject'],
            'qg_source_id': qg_edge['subject'],
            'qg_target_id': qg_edge['object']
        }

        # If single value provided for predicate: make it an array
        if not isinstance(query_plan[edge_id]['predicate'], list):
            query_plan[edge_id]['predicate'] = [ query_plan[edge_id]['predicate'] ]

        # Get the nodes infos in the query plan object
        for node_id, node in query_graph["nodes"].items():
        # for node in query_graph['nodes']:
            if node_id == qg_edge['subject'] or node_id == qg_edge['object']:
                if 'id' in node:
                    # If single values provided for id or category: make it an array
                    if not isinstance(node['id'], list):
                        node['id'] = [ node['id'] ]
                    # Resolve the curie provided with the NodeNormalization API
                    query_plan[edge_id]['from_kg_id'], resolved_ids_object = resolve_ids_with_nodenormalization_api(node['id'], resolved_ids_object)
                    query_plan[edge_id]['from_qg_id'] = node_id
                    query_plan[edge_id]['from_type'] = node['category']
                    if not isinstance(query_plan[edge_id]['from_type'], list):
                        query_plan[edge_id]['from_type'] = [ query_plan[edge_id]['from_type'] ]

                else:
                    # The node without curie is the association's "to"
                    query_plan[edge_id]['to_qg_id'] = node_id
                    query_plan[edge_id]['to_type'] = node['category']
                    if not isinstance(query_plan[edge_id]['to_type'], list):
                        query_plan[edge_id]['to_type'] = [ query_plan[edge_id]['to_type'] ]

    knowledge_graph = {'nodes': {}, 'edges': {}}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan.keys():
        
        # DIRTY: only query if the predicate is biolink:treats or biolink:treated_by
        if 'biolink:treats' in query_plan[edge_qg_id]['predicate'] or 'biolink:treated_by' in query_plan[edge_qg_id]['predicate']:
            
            # Iterate over the list of ids provided
            for id_to_predict in query_plan[edge_qg_id]['from_kg_id']:
                try:
                    # Run OpenPredict to get predictions
                    bte_response, prediction_json = get_predictions(id_to_predict, model_id, min_score, max_score)
                except:
                    # except KeyError: ?
                    prediction_json = []
                for association in prediction_json:
                    # TODO:if the target IDs is given, filter here 
                    
                    edge_kg_id = 'e' + str(kg_edge_count)
                    # Get the ID of the predicted entity in result association
                    # based on the type expected for the association "to" node
                    # node_dict[query_plan[edge_qg_id]['from_kg_id']] = query_plan[edge_qg_id]['from_type']
                    # node_dict[association[query_plan[edge_qg_id]['to_type']]] = query_plan[edge_qg_id]['to_type']

                    # id/type of nodes are registered in a dict to avoid duplicate in knowledge_graph.nodes
                    # Build dict of node ID : label
                    source_node_id = resolve_id(association['source']['id'], resolved_ids_object)
                    target_node_id = resolve_id(association['target']['id'], resolved_ids_object)
                    node_dict[source_node_id] = {
                        'type': association['source']['type']
                    }
                    if 'label' in association['source'] and association['source']['label']:
                        node_dict[source_node_id]['label'] = association['source']['label']

                    node_dict[target_node_id] = {
                        'type': association['target']['type']
                    }
                    if 'label' in association['target'] and association['target']['label']:
                        node_dict[target_node_id]['label'] = association['target']['label']

                    # edge_association_type = 'biolink:ChemicalToDiseaseOrPhenotypicFeatureAssociation'
                    source = 'OpenPredict'
                    relation = 'RO:0002434'
                    # relation = 'OBOREL:0002606'
                    association_score = str(association['score'])

                    # See attributes examples: https://github.com/NCATSTranslator/Evidence-Provenance-Confidence-Working-Group/blob/master/attribute_epc_examples/COHD_TRAPI1.1_Attribute_Example_2-3-21.yml
                    edge_dict = {
                        # TODO: not required anymore? 'association_type': edge_association_type,
                        'relation': relation,
                        'attributes': [
                            {
                                "name": "model_id",
                                "source": source,
                                "type": "EDAM:data_1048",
                                "value": model_id
                            },
                            {
                                # TODO: use has_confidence_level?
                                "name": "score",
                                "source": source,
                                "type": "EDAM:data_1772",
                                "value": association_score
                                # https://www.ebi.ac.uk/ols/ontologies/edam/terms?iri=http%3A%2F%2Fedamontology.org%2Fdata_1772&viewMode=All&siblings=false
                            },
                        ]
                    }

                    # Map the source/target of query_graph to source/target of association
                    # if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                    edge_dict['subject'] = source_node_id
                    edge_dict['object'] = target_node_id
                    if association['source']['type'] == 'drug':
                        # and 'biolink:Drug' in query_plan[edge_qg_id]['predicate']: ?
                        edge_dict['predicate'] = 'biolink:treats'
                    else: 
                        edge_dict['predicate'] = 'biolink:treated_by'

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
            prediction_json = []

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_to_add = {
            'category': 'biolink:' + properties['type'].capitalize() ,
            }
        if 'label' in properties and properties['label']:
            node_to_add['name'] = properties['label']

        knowledge_graph['nodes'][node_id] = node_to_add
    return {"message": {'knowledge_graph': knowledge_graph, 'query_graph': query_graph, 'results': query_results}}


# TOREMOVE: Example of TRAPI queries
simple_json = {
  "message": {
    "query_graph": {
      "edges": {
        "e01": {
          "object": "n1",
          "predicate": "biolink:treated_by",
          "subject": "n0"
        }
      },
      "nodes": {
        "n0": {
          "category": "biolink:Disease",
          "id": "OMIM:246300"
        },
        "n1": {
          "category": "biolink:Drug"
        }
      }
    }
  },
  "query_options": {
    "max_score": 1,
    "min_score": 0.5,
    "n_results": 2
  }
}

array_json = {
  "message": {
    "query_graph": {
      "edges": {
        "e01": {
          "object": "n1",
          "predicate": ["biolink:treated_by", "biolink:treats"],
          "subject": "n0"
        }
      },
      "nodes": {
        "n0": {
          "category": ["biolink:Disease", "biolink:Drug"],
          "id": ["OMIM:246300", "DRUGBANK:DB00394"]
        },
        "n1": {
          "category": ["biolink:Drug", "biolink:Disease"]
        }
      }
    }
  },
  "query_options": {
    "max_score": 1,
    "min_score": 0.5
  }
}