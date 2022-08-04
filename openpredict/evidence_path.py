import ast

import networkx as nx
import pandas as pd
import pkg_resources
from gensim.models import KeyedVectors

#from openpredict.utils import get_openpredict_dir

# Access uncommitted data in the persistent data directory
# get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')

# Access the openpredict/data folder for data that has been committed
# pkg_resources.resource_filename('openpredict', 'data/features/openpredict-baseline-omim-drugbank.joblib')


df_op = pd.read_csv("openpredict/data/resources/openpredict-omim-drug.csv")
drug_fp_vectors = KeyedVectors.load_word2vec_format(
    'openpredict/data/embedding/drugs_fp_embed.txt', binary=False)
disease_hp_vectors = KeyedVectors.load_word2vec_format(
    'openpredict/data/embedding/disease_hp_embed.txt', binary=False)


df_op = df_op.rename(columns={'omimid': 'disease_id', 'drugid': 'drug_id'})
df_op.disease_id = df_op.disease_id.astype(str)



indications_dict = set()
for i, row in df_op.iterrows():
    #row['DB_ID'], row['DO_ID']
    pair = (str(row['drug_id']), str(row['disease_id']))
    indications_dict.add(pair)



def generate_paths_for_apair(drug, disease, drug_emb_vectors, disease_emb_vectors):
    g = nx.Graph()
    #g.add_edge(DRUGB[dr], OMIM[ds], weight=1)
    similarDrugs = drug_emb_vectors.most_similar(drug, topn=100)
    #print (similarDrugs)
    g.add_node("DrugBank:"+drug, id="DrugBank:"+drug,
               name="fake", categories=["biolink:Drug"])
    for dr, sim in similarDrugs:
        #print(dr, ' ', drug, 1-sim)
        #g.add_edge(dr, drug, weight=1-sim)

        g.add_node("DrugBank:"+dr, id="DrugBank:"+dr,
                   name="fake", categories=["biolink:Drug"])
        g.add_edge("DrugBank:"+dr, "DrugBank:"+drug, id="DrugBank:"+dr+"_DrugBank: "+drug,
                   predicate="biolink:similar_to", subject="DrugBank:"+dr, object="DrugBank:"+drug,  weight=1-sim, attributes={"description": "score",
                                                               "attribute_type_id": "EDAM:data_1772",
                                                               "value": 1+(1-sim)
                                                               })
    g.add_node("OMIM:"+disease, id="OMIM:"+disease,
               name="fake", categories=["biolink:Disease"])
               
    similarDiseases = disease_emb_vectors.most_similar(disease, topn=100)
    for ds, sim in similarDiseases:
        #print(ds, ' ', disease)
        #g.add_edge(ds, disease, weight=1-sim)
        g.add_node("OMIM:"+ds, id="OMIM:"+ds,
                   name="fake", categories=["biolink:Disease"])
        g.add_edge("OMIM:"+ds, "OMIM:"+disease,
                   id="OMIM:" + ds+"_OMIM:"+disease, predicate="biolink:similar_to", subject="OMIM:"+ds, object="OMIM:"+disease, weight=1-sim, attributes={"description": "score",
                                                                                                 "attribute_type_id": "EDAM:data_1772",
                                                                                                 "value": 1+(1-sim)
                                                                                                 })
                                                                                                 
    for (dr, ds) in indications_dict:
        if "DrugBank:"+dr in g.nodes() and "OMIM:"+ds in g.nodes():
            #print(dr, ds, 1)
            #g.add_edge(dr, ds, weight=1)
            g.add_edge("DrugBank:"+dr, "OMIM:"+ds, id="DrugBank:" +
                       dr+"_OMIM:"+ds, predicate="biolink:treats", subject="DrugBank:"+dr, object="OMIM:"+ds,  weight=1.0, attributes={"description": "score",
                                                                               "attribute_type_id": "EDAM:data_1772",
                                                                               "value": "1.0"
                                                                               })

    
    return g




def generate_explanation(drug, disease, drug_fp_vectors, disease_hp_vectors, topK):
    #-> Path generation, add similar_to relation between query drug and disease
    #-> add known treats relations if any drug-disease pair in the graph has a treats relation
    g1 = generate_paths_for_apair(
        drug, disease, drug_fp_vectors, disease_hp_vectors)
    # Iterate over all simple paths
    # assign a weight to each path by summing their weights (for similar_to weight is 1-similarity, for treats, weight is 1)
    path_weight = {}
    for path in nx.all_simple_paths(g1,"DrugBank:"+drug,"OMIM:"+disease, cutoff=4):
        dpath = 0
        for i in range(len(path)-1):
            #print(g1[path[i]][path[i+1]])
            dpath += g1[path[i]][path[i+1]]['weight']
        path_weight[str(path)] = dpath
    # rank the paths and take only top-K paths
    path_weight_dict = sorted(path_weight.items(), key=lambda x: x[1], )
    # create a final graph by merging the top-K paths
    G = nx.Graph()
    for p, s in path_weight_dict[:topK]:
        path = ast.literal_eval(p)
        for i in range(len(path)-1):
            s_node_name = path[i]
            t_node_name = path[i+1]
            edge_data = g1[s_node_name][t_node_name]

            G.add_node(s_node_name, id="DrugBank:"+drug,
               name="fake", categories=["biolink:Drug"])
            G.add_node(t_node_name, id="DrugBank:"+drug,
               name="fake", categories=["biolink:Drug"])
        
            G.add_edge(s_node_name, t_node_name, id = edge_data["id"], predicate= edge_data["predicate"], 
            subject = edge_data["subject"], object = edge_data["object"], weight=edge_data["weight"],
            attributes= edge_data["attributes"])
   
    return G


 
def generate_json(graph) : 
    graph_json ={}
    graph_json['nodes'] = list()
    
    for node in graph.nodes():
        graph_json['nodes'].append(graph[node])
   
    graph_json['edges']=list()
    for edge in graph.edges():
        graph_json['edges'].append(graph[edge[0]][edge[1]])
   
    return graph_json



def do_evidence_path(drug_id: str, disease_id: str, topK : int):

    evidence_path = generate_explanation(drug=drug_id, disease=disease_id, drug_fp_vectors = drug_fp_vectors, disease_hp_vectors= disease_hp_vectors,topK=topK)
    
    return generate_json(evidence_path)
 