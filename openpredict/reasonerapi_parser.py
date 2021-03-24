from openpredict.openpredict_model import get_predictions

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
            'qg_source_id': qg_edge['subject'],
            'qg_target_id': qg_edge['object']
        }

        # Get the nodes infos in the query plan object
        for node_id, node in query_graph["nodes"].items():
        # for node in query_graph['nodes']:
            if node_id == qg_edge['subject'] or node_id == qg_edge['object']:
                if 'id' in node:
                    # The node with curie is the association's "from"
                    query_plan[edge_id]['from_kg_id'] = node['id']
                    query_plan[edge_id]['from_qg_id'] = node_id
                    query_plan[edge_id]['from_type'] = node['category'].lower()
                else:
                    # The node without curie is the association's "to"
                    query_plan[edge_id]['to_qg_id'] = node_id
                    query_plan[edge_id]['to_type'] = node['category'].lower()

        # TODO: edge type should be required

    knowledge_graph = {'nodes': {}, 'edges': {}}
    node_dict = {}
    query_results = []
    kg_edge_count = 0

    # Now iterates the query plan to execute each query
    for edge_qg_id in query_plan.keys():
        
        # Run get_predict!
        # TODO: pass score and limit from Reasoner query
        # TODO: add try catch and n_results
        
        if query_plan[edge_qg_id]['predicate'] == 'biolink:treats' or query_plan[edge_qg_id]['predicate'] == 'biolink:treated_by':
            bte_response, prediction_json = get_predictions(query_plan[edge_qg_id]['from_kg_id'], model_id, min_score, max_score)
        else: 
            prediction_json = []

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

            # TODO: make it dynamic?
            # edge_association_type = 'biolink:ChemicalToDiseaseOrPhenotypicFeatureAssociation'
            source = 'OpenPredict'
            relation = 'RO:0002434'
            # relation = 'OBOREL:0002606'
            association_score = str(association['score'])   

            # See attributes examples: https://github.com/NCATSTranslator/Evidence-Provenance-Confidence-Working-Group/blob/master/attribute_epc_examples/COHD_TRAPI1.1_Attribute_Example_2-3-21.yml
            edge_dict = {
                # TODO: not required anymore? 'association_type': edge_association_type,
                'predicate': query_plan[edge_qg_id]['predicate'],
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
            if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                edge_dict['subject'] = association['source']['id']
                edge_dict['object'] = association['target']['id']    
            else: 
                edge_dict['subject'] = association['target']['id']
                edge_dict['object'] = association['source']['id']    

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


    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_to_add = {
            'category': 'biolink:' + properties['type'].capitalize() ,
            }
        if 'label' in properties and properties['label']:
            node_to_add['name'] = properties['label']

        knowledge_graph['nodes'][node_id] = node_to_add
    return {"message": {'knowledge_graph': knowledge_graph, 'query_graph': query_graph, 'results': query_results}}
