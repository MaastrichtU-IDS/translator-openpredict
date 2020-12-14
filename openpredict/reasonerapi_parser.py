from openpredict.openpredict_model import get_predictions

def typed_results_to_reasonerapi(reasoner_query, model_id):
    """Convert an array of predictions objects to ReasonerAPI format
    Run the get_predict to get the QueryGraph edges and nodes
    {disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

    :param: reasoner_query Query from Reasoner API
    :return: Results as ReasonerAPI object
    """
    query_graph = reasoner_query["message"]["query_graph"]
    try:
        min_score = float(reasoner_query["message"]["query_options"]["min_score"])
    except:
        print('min score retrieve failed')
        min_score=None
    try:
        max_score = float(reasoner_query["message"]["query_options"]["max_score"])
    except: 
        print('max score retrieve failed')
        max_score=None
    query_plan = {}
    # Parse the query_graph to build the query plan
    for qg_edge in query_graph["edges"]:
        # Build dict with all infos of associations to predict 
        query_plan[qg_edge['id']] = {
            'association_type': qg_edge['type'],
            'qg_source_id': qg_edge['source_id'],
            'qg_target_id': qg_edge['target_id']
        }

        # Get the nodes infos in the query plan object
        for node in query_graph['nodes']:
            if node['id'] == qg_edge['source_id'] or node['id'] == qg_edge['target_id']:
                if 'curie' in node:
                    # The node with curie is the association's "from"
                    query_plan[qg_edge['id']]['from_kg_id'] = node['curie']
                    query_plan[qg_edge['id']]['from_qg_id'] = node['id']
                    query_plan[qg_edge['id']]['from_type'] = node['type'].lower()
                else:
                    # The node without curie is the association's "to"
                    query_plan[qg_edge['id']]['to_qg_id'] = node['id']
                    query_plan[qg_edge['id']]['to_type'] = node['type'].lower()

        # TODO: edge type should be required

    knowledge_graph = {'nodes': [], 'edges': []}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan.keys():
        
        # Run get_predict!
        # TODO: pass score and limit from Reasoner query
        # TODO: add try catch and n_results
        bte_response, prediction_json = get_predictions(query_plan[edge_qg_id]['from_kg_id'], model_id, min_score, max_score)

        for association in prediction_json:
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

            edge_dict = {
                'id': edge_kg_id,
                'type': query_plan[edge_qg_id]['association_type'],
                'score': association['score'] }

            # Map the source/target of query_graph to source/target of association
            if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                edge_dict['source_id'] = association['source']['id']
                edge_dict['target_id'] = association['target']['id']    
            else: 
                edge_dict['source_id'] = association['target']['id']
                edge_dict['target_id'] = association['source']['id']    

            # Add the association in the knowledge_graph as edge
            # Use the type as key in the result association dict (for IDs)
            knowledge_graph['edges'].append(edge_dict)

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
                    "kg_id": association['source']['id'],
                    'qg_id': query_plan[edge_qg_id]['from_qg_id']
                })
            result['node_bindings'].append(
                {
                    "kg_id": association['target']['id'],
                    'qg_id': query_plan[edge_qg_id]['to_qg_id']
                })
            query_results.append(result)
            kg_edge_count += 1

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_to_add = {
            'id': node_id,
            'type': properties['type'],
            }
        if 'label' in properties and properties['label']:
            node_to_add['name'] = properties['label']

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