from openpredict.openpredict_model import get_predictions
import requests
import json


def is_accepted_id(id_to_check):
    if id_to_check.lower().startswith('omim') or id_to_check.lower().startswith('drugbank'):
        return True
    else:
        return False

def resolve_ids_with_nodenormalization_api(resolve_ids_list):
    max_predictions_returned = 5
    resolved_ids_list = []
    ids_to_normalize = []

    for id_to_resolve in resolve_ids_list:
        if is_accepted_id(id_to_resolve):
            resolved_ids_list.append(id_to_resolve)
        else:
            ids_to_normalize.append(id_to_resolve)
    
    # mondo_ids_list = ["MONDO:0018874", "MONDO:0008734", "MONDO:0004056", "MONDO:0005499", "MONDO:0006256", "MONDO:0006143", "MONDO:0019087", "MONDO:0002271", "MONDO:0003093", "MONDO:0018177", "MONDO:0010150", "MONDO:0017885", "MONDO:0005005", "MONDO:0017884", "MONDO:0007256", "MONDO:0005061", "MONDO:0005097", "MONDO:0018905", "MONDO:0005065", "MONDO:0006046", "MONDO:0006047", "MONDO:0004974", "MONDO:0005082", "MONDO:0002169", "MONDO:0005089", "MONDO:0005012", "MONDO:0005036", "MONDO:0010108", "MONDO:0006456", "MONDO:0015075", "MONDO:0006485", "MONDO:0000553", "MONDO:0006486", "MONDO:0004967", "MONDO:0005170", "MONDO:0005072", "MONDO:0008433", "MONDO:0004163", "MONDO:0000554", "MONDO:0005580", "MONDO:0004093", "MONDO:0000448"]
    # OMIM:246300
    # First query Translator NodeNormalization API to convert MONDO IDs to OMIM IDs
    if len(ids_to_normalize) > 0:
        resolve_curies = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                            params={'curie': ids_to_normalize})

        print('RESOLVED')
        print(resolve_curies.content)
        # Get corresponding OMIM IDs for MONDO IDs if match
        resp = resolve_curies.json()
        for preferred_id, alt_ids in resp.items():
            for alt_id in alt_ids['equivalent_identifiers']:
                if is_accepted_id(str(alt_id['identifier'])):
                    resolved_ids_list.append(str(alt_id['identifier']))
                    print('🗺 Mapped ' + preferred_id + ' - "' + alt_ids['id']['label'] + '" to ' + alt_id['identifier'])
                
    return resolved_ids_list

def typed_results_to_reasonerapi(reasoner_query):
    """Convert an array of predictions objects to ReasonerAPI format
    Run the get_predict to get the QueryGraph edges and nodes
    {disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

    :param: reasoner_query Query from Reasoner API
    :return: Results as ReasonerAPI object
    """
    # TODO: note that the implementation is dirty with a lot of hardcoded tweaks.
    # This is due to the ever changing nature of the Translator ecosystem
    # There are no standards so all the reaoning and parsing system needs to be quickly implemented with no docs.

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
                    # If IDs are not OMIM or DrugBank we use the NodeNormalization API
                    # for node_id_to_check in node['id']:
                    #     if not node_id_to_check.lower().startwith('omim') and not node_id_to_check.lower().startwith('drugbank'):
                            # Call SRI API to resolve

                    # If single values provided for id or category: make it an array
                    if not isinstance(node['id'], list):
                        node['id'] = [ node['id'] ]
                    # The node with curie is the association "from"
                    query_plan[edge_id]['from_kg_id'] = resolve_ids_with_nodenormalization_api(node['id'])
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

        # TODO: edge type should be required?

    knowledge_graph = {'nodes': {}, 'edges': {}}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    print('QUERY PLAN:')
    print(query_plan)

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan.keys():
        
        # TODO: Handle when single value OR array:
        # QEdge predicate
        # QNode id and category
        # Node category

        # DIRTY: only query if the predicate is biolink:treats or biolink:treated_by
        if 'biolink:treats' in query_plan[edge_qg_id]['predicate'] or 'biolink:treated_by' in query_plan[edge_qg_id]['predicate']:
            
            # Iterate over the list of ids provided
            for id_to_predict in query_plan[edge_qg_id]['from_kg_id']:
                print('ID to predict:')
                print(id_to_predict)
                try:
                    # Run OpenPredict to get predictions
                    bte_response, prediction_json = get_predictions(id_to_predict, model_id, min_score, max_score)
                except:
                    # except KeyError: ?
                    prediction_json = []
                print('PREDICTION JSON!')
                print(prediction_json)
                for association in prediction_json:
                    # TODO:if the target IDs is given, filter here 
                    
                    edge_kg_id = 'e' + str(kg_edge_count)
                    # Get the ID of the predicted entity in result association
                    # based on the type expected for the association "to" node
                    # node_dict[query_plan[edge_qg_id]['from_kg_id']] = query_plan[edge_qg_id]['from_type']
                    # node_dict[association[query_plan[edge_qg_id]['to_type']]] = query_plan[edge_qg_id]['to_type']
                    
                    # id/type of nodes are registered in a dict to avoid duplicate in knowledge_graph.nodes
                    # Build dict of node ID : label
                    node_dict[association['source']['id']] = {
                        'type': association['source']['type']
                    }
                    if 'label' in association['source'] and association['source']['label']:
                        node_dict[association['source']['id']]['label'] = association['source']['label']

                    node_dict[association['target']['id']] = {
                        'type': association['target']['type']
                    }
                    if 'label' in association['target'] and association['target']['label']:
                        node_dict[association['target']['id']]['label'] = association['target']['label']

                    # TODO: make it dynamic?
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
                    edge_dict['subject'] = association['source']['id']
                    edge_dict['object'] = association['target']['id']
                    if association['source']['type'] == 'drug':
                        # and 'biolink:Drug' in query_plan[edge_qg_id]['predicate']: ?
                        edge_dict['predicate'] = 'biolink:treats'
                    else: 
                        edge_dict['predicate'] = 'biolink:treated_by'

                    # if edge_dict['object'].lower().startswith('drugbank'):
                    # else: 
                    #     edge_dict['subject'] = association['target']['id']
                    #     edge_dict['object'] = association['source']['id']    

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
                            "id": association['source']['id']
                        }
                    ]
                    result['node_bindings'][query_plan[edge_qg_id]['to_qg_id']] = [
                        {
                            "id": association['target']['id']
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


# Example

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