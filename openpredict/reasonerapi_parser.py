from openpredict.openpredict_model import get_predictions, get_similarities, load_similarity_embedding_models
import requests
import os
import re


def is_accepted_id(id_to_check):
    if id_to_check.lower().startswith('omim') or id_to_check.lower().startswith('drugbank'):
        return True
    else:
        return False

biolinkVersion = os.getenv('BIOLINK_VERSION', '2.2.3')


def get_biolink_parents(concept):
    concept_snakecase = concept.replace('biolink:', '')
    concept_snakecase = re.sub(r'(?<!^)(?=[A-Z])', '_', concept_snakecase).lower()
    try:
        resolve_curies = requests.get('https://bl-lookup-sri.renci.org/bl/' + concept_snakecase + '/ancestors',
                            params={'version': biolinkVersion})
        resp = resolve_curies.json()
        resp.append(concept)
        return resp
    except Exception as e:
        print('Error querying https://bl-lookup-sri.renci.org, using the original IDs')
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
        except Exception as e:
            print('Error querying the NodeNormalization API, using the original IDs')

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
        
        if 'from_kg_id' in query_plan[edge_qg_id]:
            # DIRTY: only query if the predicate is biolink:treats or biolink:treated_by or biolink:ameliorates
            # if 'biolink:treats' in query_plan[edge_qg_id]['predicates'] or 'biolink:ameliorates' in query_plan[edge_qg_id]['predicates'] or 'biolink:treated_by' in query_plan[edge_qg_id]['predicates'] or 'biolink:related_to' in query_plan[edge_qg_id]['predicates']:
            
            predicate_parents = get_biolink_parents('biolink:similar_to')
            if any(i in predicate_parents for i in query_plan[edge_qg_id]['predicates']):
                if not all_emb_vectors or all_emb_vectors == {}:
                    all_emb_vectors = load_similarity_embedding_models()

                try:
                    emb_vectors = all_emb_vectors[model_id]
                    similarity_json = get_similarities(
                        query_plan[edge_qg_id]['from_type'],
                        query_plan[edge_qg_id]['from_kg_id'], 
                        emb_vectors, min_score, max_score, n_results
                    )

                # {
                #   "count": 508,
                #   "hits": [
                #     {
                #       "id": "DRUGBANK:DB00390",
                #       "label": "Digoxin",
                #       "score": 0.9826133251190186,
                #       "type": "drug"
                #     },
                #     {
                #       "id": "DRUGBANK:DB00396",
                #       "label": "Progesterone",
                #       "score": 0.9735659956932068,
                #       "type": "drug"
                #     },

                    for hit in similarity_json['hits']:
                        source_node_id = resolve_id(query_plan[edge_qg_id]['from_kg_id'], resolved_ids_object)
                        target_node_id = resolve_id(hit['id'], resolved_ids_object)

                        node_dict[source_node_id] = {
                            'type': query_plan[edge_qg_id]['from_type']
                        }
                        node_dict[target_node_id] = {
                            'type': hit['type']
                        }

                        if 'label' in hit.keys():
                            node_dict[target_node_id]['label'] = hit['label']

                        edge_kg_id = 'e' + str(kg_edge_count)

                        association_score = str(hit['score'])

                        # See attributes examples: https://github.com/NCATSTranslator/Evidence-Provenance-Confidence-Working-Group/blob/master/attribute_epc_examples/COHD_TRAPI1.1_Attribute_Example_2-3-21.yml
                        edge_dict = {
                            # TODO: not required anymore? 'association_type': edge_association_type,
                            'relation': relation,

                            # More details on attributes: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#attribute-
                            'attributes': [
                                {
                                    "description": "model_id",
                                    "attribute_type_id": "EDAM:data_1048",
                                    "value": model_id
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
                                    'value_url': 'https://openpredict.semanticscience.org/query'
                                },
                                {
                                    'attribute_type_id': 'biolink:supporting_data_source',
                                    'value': 'infores:cohd',
                                    'value_type_id': 'biolink:InformationResource',
                                    'attribute_source': 'infores:openpredict',
                                    'value_url': 'https://openpredict.semanticscience.org'
                                },
                            ]
                        }
                        edge_dict['subject'] = source_node_id
                        edge_dict['object'] = target_node_id
                        edge_dict['predicate'] = 'biolink:similar_to'

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



                    prediction_json = []
                    print('SIMILARITY DONE')
                    print(similarity_json)
                except Exception as e:
                    print('Error processing ID ' + query_plan[edge_qg_id]['from_kg_id'])
                    print(e)
                    return ('Not found: entry in OpenPredict for ID ' + query_plan[edge_qg_id]['from_kg_id'], 404)




            predicate_parents = get_biolink_parents('biolink:treats') + get_biolink_parents('biolink:treated_by')
            if any(i in predicate_parents for i in query_plan[edge_qg_id]['predicates']):
                # Resolve when asking for treats prediction
                drugdisease_parents = get_biolink_parents('biolink:Drug') + get_biolink_parents('biolink:Disease')
                if any(i in drugdisease_parents for i in query_plan[edge_qg_id]['from_type']) and any(i in drugdisease_parents for i in query_plan[edge_qg_id]['to_type']):

                    # Iterate over the list of ids provided
                    for id_to_predict in query_plan[edge_qg_id]['from_kg_id']:
                        try:
                            # Run OpenPredict to get predictions
                            bte_response, prediction_json = get_predictions(id_to_predict, model_id, min_score, max_score)
                        except:
                            prediction_json = []
                            
                        for association in prediction_json:
                            # id/type of nodes are registered in a dict to avoid duplicate in knowledge_graph.nodes
                            # Build dict of node ID : label
                            source_node_id = resolve_id(association['source']['id'], resolved_ids_object)
                            target_node_id = resolve_id(association['target']['id'], resolved_ids_object)

                            # If the target ID is given, we filter here from the predictions
                            if 'to_kg_id' in query_plan[edge_qg_id] and target_node_id not in query_plan[edge_qg_id]['to_kg_id']:
                                pass
                            
                            else:
                                edge_kg_id = 'e' + str(kg_edge_count)
                                # Get the ID of the predicted entity in result association
                                # based on the type expected for the association "to" node
                                # node_dict[query_plan[edge_qg_id]['from_kg_id']] = query_plan[edge_qg_id]['from_type']
                                # node_dict[association[query_plan[edge_qg_id]['to_type']]] = query_plan[edge_qg_id]['to_type']
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

                                    # More details on attributes: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#attribute-
                                    'attributes': [
                                        {
                                            "description": "model_id",
                                            "attribute_type_id": "EDAM:data_1048",
                                            "value": model_id
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
                                            'value_url': 'https://openpredict.semanticscience.org/query'
                                        },
                                        {
                                            'attribute_type_id': 'biolink:supporting_data_source',
                                            'value': 'infores:cohd',
                                            'value_type_id': 'biolink:InformationResource',
                                            'attribute_source': 'infores:openpredict',
                                            'value_url': 'https://openpredict.semanticscience.org'
                                        },
                                    ]
                                }

                                # Map the source/target of query_graph to source/target of association
                                # if association['source']['type'] == query_plan[edge_qg_id]['from_type']:
                                edge_dict['subject'] = source_node_id
                                edge_dict['object'] = target_node_id

                                # Define the predicate depending on the association source type returned by OpenPredict classifier
                                if association['source']['type'] == 'drug':
                                    # and 'biolink:Drug' in query_plan[edge_qg_id]['predicates']: ?
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
                    print('BioLink category not parents of Drug or Disease, no results returned')
                    prediction_json = []

        else: 
            prediction_json = []

    # Generate kg nodes from the dict of nodes + result from query to resolve labels
    for node_id, properties in node_dict.items():
        node_to_add = {
            'categories': ['biolink:' + properties['type'].capitalize()] ,
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