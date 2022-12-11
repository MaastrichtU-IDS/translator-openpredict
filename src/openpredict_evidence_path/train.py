import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from openpredict.config import settings
from openpredict_model.utils import load_treatment_embeddings

## Evidence path for OpenPredict model by Elif


drug_fp_vectors = KeyedVectors.load_word2vec_format(
    f'{settings.OPENPREDICT_DATA_DIR}/embedding/drugs_fp_embed.txt', binary=False)
disease_hp_vectors = KeyedVectors.load_word2vec_format(
    f'{settings.OPENPREDICT_DATA_DIR}/embedding/disease_hp_embed.txt', binary=False)


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
        'data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDrugs.' + str(feature) + '.txt', binary=False)
        calculateEntitySimilarities(drug_emb)
        dr,ds = getQuantiles(drug_emb, disease_hp_vectors,1)
        feature_percentiles[feature] = dr

    for feature in features_diseases :
        disease_emb = KeyedVectors.load_word2vec_format(
        'data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDiseases.' + str(feature) + '.txt', binary=False)
        calculateEntitySimilarities(disease_emb)
        dr,ds = getQuantiles(drug_fp_vectors, disease_emb,0.25)
        feature_percentiles[feature] = ds

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
    np.savetxt('data/evidence-path-model/feature_' + fileName, embedding_df_np, fmt = '%f' )
