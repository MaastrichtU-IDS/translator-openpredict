import ast
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from openpredict.config import settings
from openpredict.models import openpredict_model
from openpredict.utils import load_treatment_embeddings

## Evidence path for OpenPredict model by Elif

#from openpredict_model import load_treatment_embeddings

#from openpredict.utils import get_openpredict_dir

# Access uncommitted data in the persistent data directory
# get_openpredict_dir('features/openpredict-baseline-omim-drugbank.joblib')

# Access the openpredict/data folder for data that has been committed
# 'openpredict/data/features/openpredict-baseline-omim-drugbank.joblib'

# class ModelEvidencePath():


# MODEL_DIR_PATH = settings.OPENPREDICT_DATA_DIR + '/evidence-path-model'
# DOWNLOAD_MODEL_URL = 'https://download.dumontierlab.com/openpredict-models/evidence-path-model.zip'

# def init():
#     print('Initializing Evidence path')
#     if not os.path.exists(f"{MODEL_DIR_PATH}"):
#         print("📥️ Evidence path model not present, downloading it")
#         Path(f"{MODEL_DIR_PATH}").mkdir(parents=True, exist_ok=True)
#         os.system(f"wget -O {MODEL_DIR_PATH}/evidence-path-model.zip {DOWNLOAD_MODEL_URL}")
#         os.system(f'unzip "{MODEL_DIR_PATH}/*.zip" -d {settings.OPENPREDICT_DATA_DIR}')
#         os.system(f"rm {MODEL_DIR_PATH}/*.zip")
#     else:
#         print("✅ Model already present")

# df_op = pd.read_csv("openpredict/data/resources/openpredict-omim-drug.csv")


df_op = pd.read_csv(f"{settings.GIT_DATA_DIR}/resources/openpredict-omim-drug.csv")

drug_fp_vectors = KeyedVectors.load_word2vec_format(
    f'{settings.GIT_DATA_DIR}/embedding/drugs_fp_embed.txt', binary=False)
disease_hp_vectors = KeyedVectors.load_word2vec_format(
    f'{settings.GIT_DATA_DIR}/embedding/disease_hp_embed.txt', binary=False)

df_op = df_op.rename(columns={'omimid': 'disease_id', 'drugid': 'drug_id'})
df_op.disease_id = df_op.disease_id.astype(str)

indications_dict = set()
for i, row in df_op.iterrows():
    #row['DB_ID'], row['DO_ID']
    pair = (str(row['drug_id']), str(row['disease_id']))
    indications_dict.add(pair)


#functions which are used to generate the evidence path
def generate_paths_for_apair(drug, disease, drug_emb_vectors, disease_emb_vectors,features_drug = None, features_disease = None,threshold_drugs = 1,threshold_diseases = 1):
    g = nx.Graph()
    (threshold_drug,threshold_disease) =getQuantiles(drug_emb_vectors, disease_emb_vectors, threshold_drugs)

    if(features_drug is not None) :
         filtered_embedding_drugs = KeyedVectors.load_word2vec_format(f'{settings.OPENPREDICT_DATA_DIR}/evidence-path-model/feature_{str(features_drug)}.txt', binary=False)
         similarDrugs = filtered_embedding_drugs.most_similar(drug, topn=100)
         (threshold_drug,threshold_disease) =getQuantiles(filtered_embedding_drugs, disease_emb_vectors, threshold_drugs)
    else :
       similarDrugs = drug_emb_vectors.most_similar(drug, topn=100)


    g.add_node("DRUGBANK:"+drug, id="DRUGBANK:"+drug,
               name="fake", categories=["biolink:Drug"])
    for dr, sim in similarDrugs:
        if ((1-sim) <= threshold_drug) :
             g.add_node("DRUGBANK:"+dr, id="DRUGBANK:"+dr,
                   name="fake", categories=["biolink:Drug"])
             g.add_edge("DRUGBANK:"+dr, "DRUGBANK:"+drug, id="DRUGBANK:"+dr+"_DRUGBANK: "+drug,
                   predicate="biolink:similar_to", subject="DRUGBANK:"+dr, object="DRUGBANK:"+drug,  weight=1-sim, attributes={"description": "score",
                                                               "attribute_type_id": "EDAM:data_1772",
                                                               "value": (1-sim)                                                     })
             g.add_node("OMIM:"+disease, id="OMIM:"+disease,
               name="fake", categories=["biolink:Disease"])
    (threshold_drug,threshold_disease) =getQuantiles(drug_emb_vectors, disease_emb_vectors, threshold_diseases)


    # TODO: USE settings.OPENPREDICT_DATA_DIR instead of lucky relative path
    if(features_disease is not None) :
        filtered_embedding_diseases = KeyedVectors.load_word2vec_format(f'{settings.OPENPREDICT_DATA_DIR}/evidence-path-model/feature_{str(features_disease)}.txt', binary=False)
        # filtered_embedding_diseases = KeyedVectors.load_word2vec_format(f'openpredict/data/embedding/feature_specific_embeddings_KG/feature_{str(features_disease)}.txt', binary=False)
        similarDiseases = filtered_embedding_diseases.most_similar(disease, topn=100)
        (threshold_drug,threshold_disease) =getQuantiles(drug_fp_vectors, filtered_embedding_diseases, threshold_diseases)
    else :
        similarDiseases = disease_emb_vectors.most_similar(disease, topn=100)


    for ds, sim in similarDiseases:
        if((1-sim) <= threshold_disease) :
             g.add_node("OMIM:"+ds, id="OMIM:"+ds,
                   name="fake", categories=["biolink:Disease"])
             g.add_edge("OMIM:"+ds, "OMIM:"+disease,
                   id="OMIM:" + ds+"_OMIM:"+disease, predicate="biolink:similar_to", subject="OMIM:"+ds, object="OMIM:"+disease, weight=1-sim, attributes={"description": "score",
                                                                                                 "attribute_type_id": "EDAM:data_1772",
                                                                                                 "value": 1+(1-sim)
                                                                                                 })

    for (dr, ds) in indications_dict:
        if "DRUGBANK:"+dr in g.nodes() and "OMIM:"+ds in g.nodes():
            g.add_edge("DRUGBANK:"+dr, "OMIM:"+ds, id="DRUGBANK:" +
                         dr+"_OMIM:"+ds, predicate="biolink:treats", subject="DRUGBANK:"+dr, object="OMIM:"+ds,  weight= 1.0,
                                                                                attributes={"description": "score",
                                                                               "attribute_type_id": "EDAM:data_1772",
                                                                               "value": "1.0"
                                                                               })

    return (g)


def generate_explanation(drug, disease, drug_fp_vectors, disease_hp_vectors,features_drug = None, features_disease = None,threshold_drugs = 0,threshold_disease = 0):
    #-> Path generation, add similar_to relation between query drug and disease
    #-> add known treats relations if any drug-disease pair in the graph has a treats relation
    g1= generate_paths_for_apair(
        drug, disease, drug_fp_vectors, disease_hp_vectors,features_drug, features_disease,threshold_drugs,threshold_disease)
    # Iterate over all simple paths
    # assign a weight to each path by summing their weights (for similar_to weight is 1-similarity, for treats, weight is 1)
    path_weight = path_weight_product(g1,drug=drug, disease=disease)
    # rank the paths and take only top-K paths
    path_weight_dict = sorted(path_weight.items(), key=lambda x: x[1], )
    # create a final graph by merging the top-K paths
    G = nx.Graph()
    for p, s in path_weight_dict[:100]:
        path = ast.literal_eval(p)
        for i in range(len(path)-1):
            s_node_name = path[i]
            t_node_name = path[i+1]
            edge_data = g1[s_node_name][t_node_name]

            G.add_node(s_node_name, id="DRUGBANK:"+drug,
               name="fake", categories=["biolink:Drug"])
            G.add_node(t_node_name, id="DRUGBANK:"+drug,
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



def do_evidence_path(drug_id: str, disease_id: str, threshold_drugs : float,threshold_disease : float, features_drug, features_disease):
    ''' generates explanations based on the user input and returns a json'''
    evidence_path = generate_explanation(drug=drug_id, disease=disease_id, drug_fp_vectors = drug_fp_vectors, disease_hp_vectors= disease_hp_vectors,
                                         features_drug = features_drug, features_disease = features_disease,threshold_drugs = threshold_drugs,threshold_disease = threshold_disease )
    return generate_json(evidence_path)






###############################################
# Script used to generate embeddings
# to generate embeddings, call generate_feature_embedding_data() in method do_evidence_path()
#
def calculateEntitySimilarities(tokenized_vector, topn = 100) :
    ''' calculates similarity scores of all drug-drug and disease-disease
        pairs that exist in the knowledge base
        return : a list containing all the similarity scores '''

    entities = list(tokenized_vector.vocab)
    similarity_scores = []
    for entity in entities :
        similarEntities = tokenized_vector.most_similar(entity, topn=100)
        for ent, sim in similarEntities :
            similarity_scores.append(1-sim)

    return similarity_scores



def getQuantiles( drug_vectors, disease_vectors, quantile = 0.1) :
    ''' calulcates the nth quantile of the calculated similarity scores
        return : the min-threshold for the drugs and diseases as a tuple
    '''
    drug_similarities = calculateEntitySimilarities(drug_vectors,505)

    drug_sim_df = pd.DataFrame(drug_similarities)
    #print(drug_sim_df.describe())

    disease_similarities = calculateEntitySimilarities(disease_vectors,309)

    disease_sim_df = pd.DataFrame(disease_similarities)
    return ((drug_sim_df.quantile(quantile)[0]), (disease_sim_df.quantile(quantile)[0]))



def percentiles_of_different_features():
    features_drug = ["TC", 'PPI_SIM', 'SE_SIM', 'GO_SIM', 'TARGETSEQ_SIM']
    features_diseases = ["HPO_SIM", 'PHENO_SIM']

    feature_percentiles = dict()
    for feature in features_drug :
        drug_emb = KeyedVectors.load_word2vec_format(
        'openpredict/data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDrugs.' + str(feature) + '.txt', binary=False)
        calculateEntitySimilarities(drug_emb)
        dr,ds = getQuantiles(drug_emb, disease_hp_vectors,1)
        feature_percentiles[feature] = dr

    for feature in features_diseases :
        disease_emb = KeyedVectors.load_word2vec_format(
        'openpredict/data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDiseases.' + str(feature) + '.txt', binary=False)
        calculateEntitySimilarities(disease_emb)
        dr,ds = getQuantiles(drug_fp_vectors, disease_emb,0.25)
        feature_percentiles[feature] = ds


    print(feature_percentiles)
    return feature_percentiles


def path_weight_summation(g1,drug,disease):
    path_weight = {}
    for path in nx.all_simple_paths(g1,"DRUGBANK:"+drug,"OMIM:"+disease, cutoff=4):
        dpath = 0
        for i in range(len(path)-1):
            dpath += g1[path[i]][path[i+1]]['weight']
        path_weight[str(path)] = dpath

    return path_weight


def path_weight_product(g1,drug,disease) :
    path_weight = {}
    for path in nx.all_simple_paths(g1,"DRUGBANK:"+drug,"OMIM:"+disease, cutoff=4):
        dpath = 0
        for i in range(len(path)-1):
            dpath *= g1[path[i]][path[i+1]]['weight']
        path_weight[str(path)] = dpath

    return path_weight



def filter_out_features_diseases(features_of_interest):
    '''Creates the dataframe based on disease features to be converted to a embedding later '''
    (drug_ft_emb, disease_ft_emb) = load_treatment_embeddings('openpredict-baseline-omim-drugbank')
    resulting_embeddings = disease_ft_emb.loc[:,features_of_interest]
    #if(len(features_of_interest) > 1):
    #resulting_embeddings.columns = resulting_embeddings.columns.droplevel()
    save_embedding_as_txt(resulting_embeddings, str(features_of_interest) + ".txt")
    return resulting_embeddings

def generate_feature_embedding_data():
     drug_features = {"GO_SIM", "PPI_SIM","SE_SIM","TARGETSEQ_SIM","TC"}
     disease_features = {"HPO_SIM", "PHENO_SIM"}

     for feature in drug_features :
        df = filter_out_features_drugs(feature)
        save_embedding_as_txt(df,'feature_FeatureTypesDrugs.' +feature+ '.txt')

     for feature in disease_features :
        df = filter_out_features_diseases(feature)
        save_embedding_as_txt(df,'feature_FeatureTypesDiseases.' +feature+ '.txt')

def filter_out_features_drugs(features_of_interest) :
    '''Creates the dataframe based on drug features to be converted to a embedding later '''
    (drug_ft_emb, disease_ft_emb) = load_treatment_embeddings('openpredict-baseline-omim-drugbank')
    resulting_embeddings = drug_ft_emb.loc[:,features_of_interest]
    # if(len(features_of_interest) > 1) :
    #     resulting_embeddings.columns = resulting_embeddings.columns.droplevel()
    resulting_embeddings.index = [s.replace("DB", "") for s in list(resulting_embeddings.index.values)]
    save_embedding_as_txt(resulting_embeddings, str(features_of_interest) + ".txt")

    return resulting_embeddings


def save_embedding_as_txt(embedding_df, fileName) :
    '''
    takes the dataframe filtered based on the features and returns a txt which represents
    its embedding
    '''
    embedding_df.index = list(map(int, embedding_df.index))
    embedding_df = embedding_df.reset_index()
    embedding_df_np = embedding_df.to_numpy()
    np.savetxt('openpredict/data/embedding/feature_' + fileName, embedding_df_np, fmt = '%f' )

