import requests
import json
import logging

from openpredict.openpredict_omim_drugbank import query_omim_drugbank_classifier

def typed_results_to_reasonerapi(reasoner_query):
    """Convert an array of results of type
    {disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

    :reasoner_query Query from Reasoner API
    """
    query_graph = reasoner_query["message"]["query_graph"]
    query_plan = {}
    # Parse the query_graph to build the query plan
    for edge in query_graph["edges"]:
        # Build dict with all infos of associations to predict 
        query_plan[edge['id']] = {
            'association_type': edge['type'],
            'qg_source_id': edge['source_id'],
            'qg_target_id': edge['target_id']
        }

        # Get the nodes infos in the query plan object
        for node in query_graph['nodes']:
            if node['id'] == edge['source_id'] or node['id'] == edge['target_id']:
                if 'curie' in node:
                    # The node with curie is the association's "from"
                    query_plan[edge['id']]['from_kg_id'] = node['curie']
                    query_plan[edge['id']]['from_qg_id'] = node['id']
                    query_plan[edge['id']]['from_type'] = node['type']
                else:
                    # The node without curie is the association's "to"
                    query_plan[edge['id']]['to_qg_id'] = node['id']
                    query_plan[edge['id']]['to_type'] = node['type']

        # TODO: edge type should be required

    knowledge_graph = {'nodes': [], 'edges': []}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    # Now iterates the query plan to execute each query
    # TODO: enable passing results to enable n-hop queries
    for edge_qg_id in query_plan.keys():
        # Run get_predict!
        prediction_json = json.loads(query_omim_drugbank_classifier(query_plan[edge_qg_id]['from_kg_id']))
        # print(prediction_json)

        # Entities/type are registered in a dict to avoid duplicate in knowledge_graph.nodes
        node_dict[query_plan[edge_qg_id]['from_kg_id']] = query_plan[edge_qg_id]['from_type']
        
        for association in prediction_json:
            edge_kg_id = 'e' + str(kg_edge_count)
            # Get the ID of the predicted entity in result association
            # based on the type expected for the association "to" node
            node_dict[association[query_plan[edge_qg_id]['to_type']]] = query_plan[edge_qg_id]['to_type']

            # Add the association in the knowledge_graph as edge
            # Use the type as key in the result association dict (for IDs)
            knowledge_graph['edges'].append({
                'id': edge_kg_id,
                'source_id': association[query_plan[edge_qg_id]['from_type']],
                'target_id': association[query_plan[edge_qg_id]['to_type']],
                'type': query_plan[edge_qg_id]['association_type']
                })

            # Add the bindings to the results object
            result = {'edge_bindings': [], 'node_bindings': []}
            result['edge_bindings'].append(
                {
                    "kg_id": edge_kg_id,
                    'qg_id': edge_qg_id
                }
            )
            result['node_bindings'].append(
                {
                    "kg_id": association[query_plan[edge_qg_id]['from_type']],
                    'qg_id': query_plan[edge_qg_id]['from_qg_id']
                })
            result['node_bindings'].append(
                {
                    "kg_id": association[query_plan[edge_qg_id]['to_type']],
                    'qg_id': query_plan[edge_qg_id]['to_qg_id']
                })
            query_results.append(result)
            kg_edge_count += 1


    # Send the list of node IDs to Translator API to get labels
    # https://nodenormalization-sri.renci.org/apidocs/#/Interfaces/get_get_normalized_nodes
    # https://github.com/TranslatorIIPrototypes/NodeNormalization/blob/master/documentation/NodeNormalization.ipynb
    # TODO: add the preferred identifier to our answer?
    get_label_result = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                        params={'curie': node_dict.keys()})
    get_label_result = get_label_result.json()
    # Response is a JSON:
    # { "HP:0007354": {
    #     "id": { "identifier": "MONDO:0004976",
    #       "label": "amyotrophic lateral sclerosis" },

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id in node_dict.keys():
        node_to_add = {
            'id': node_id,
            'type': node_dict[node_id],
            }
        if node_id in get_label_result and get_label_result[node_id]:
            node_to_add['name'] = get_label_result[node_id]['id']['label']

        knowledge_graph['nodes'].append(node_to_add)
    return {'knowledge_graph': knowledge_graph, 'query_graph': query_graph, 'results': query_results}

## Sample Reasoner query
# https://github.com/broadinstitute/molecular-data-provider/blob/master/reasonerAPI/python-flask-server/openapi_server/controllers/molepro.py#L30
# https://smart-api.info/ui/912372f46127b79fb387cd2397203709#/0.9.2/post_query
# "query_graph": {
#   "edges": [
#     {
#       "id": "e00",
#       "source_id": "n00",
#       "target_id": "n01",
#       "type": "treated_by"
#     }
#   ],
#   "nodes": [
#     {
#       "curie": "MONDO:0021668",
#       "id": "n00",
#       "type": "disease"
#     },
#     {
#       "id": "n01",
#       "type": "drug"
#     }
#   ]
# }

#   "results": [
#     {
#     "edge_bindings": [
#         {
#         "kg_id": "e0",
#         "qg_id": "e00"
#         }
#     ],
#     "node_bindings": [
#         {
#         "kg_id": "MONDO:0021668",
#         "qg_id": "n00"
#         },
#         {
#         "kg_id": "ChEMBL:CHEMBL2106966",
#         "qg_id": "n01"
#         }
#     ]
#     }
#   ]