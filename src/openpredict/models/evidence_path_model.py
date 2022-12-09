import ast
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from openpredict.config import settings
from openpredict.models.base_machine_learning_model import BaseMachineLearningModel
from openpredict_model.utils import load_treatment_embeddings

# Experiment to define models as class, which download their model
# But we will probably go more towards using dvc

class EvidencePathModel(BaseMachineLearningModel):

    _folder_path: str = settings.OPENPREDICT_DATA_DIR + '/evidence-path-model'

    def __init__(self, train: bool = False):

        if os.path.exists(f"{self._folder_path}"):
            print("✅ Model already present")
        else:
            print("Otherwise: try to download pretrained model")
            try:
                self.download()
            except:
                train = True

        if train:
            print("Train the model (also if download fail/not possible)")
            self.train()

        # You can also add more objects to your model, e.g. if you need to store dataframes
        # to reuse later, for example with self.drug_fp_vectors

        self.df_op = pd.read_csv(f"{settings.OPENPREDICT_DATA_DIR}/resources/openpredict-omim-drug.csv")
        print("df_op :")
        print(self.df_op)

        self.drug_fp_vectors = KeyedVectors.load_word2vec_format(
            f'{settings.OPENPREDICT_DATA_DIR}/embedding/drugs_fp_embed.txt', binary=False)
        self.disease_hp_vectors = KeyedVectors.load_word2vec_format(
            f'{settings.OPENPREDICT_DATA_DIR}/embedding/disease_hp_embed.txt', binary=False)


        self.df_op = self.df_op.rename(columns={'omimid': 'disease_id', 'drugid': 'drug_id'})
        self.df_op.disease_id = self.df_op.disease_id.astype(str)
        (self.drug_ft_emb, self.disease_ft_emb) = load_treatment_embeddings('openpredict-baseline-omim-drugbank')

        self.indications_dict = set()
        for i, row in self.df_op.iterrows():
            #row['DB_ID'], row['DO_ID']
            pair = (str(row['drug_id']), str(row['disease_id']))
            self.indications_dict.add(pair)



    def download(self):
        """ Download and unzip pretrained model
        """
        print(f"📥️ Download and unzip pretrained model in {self._folder_path}")
        Path(f"{self._folder_path}").mkdir(parents=True, exist_ok=True)
        download_url = 'https://download.dumontierlab.com/openpredict-models/evidence-path-model.zip'
        os.system(f"wget -O {self._folder_path}/evidence-path-model.zip {download_url}")
        os.system(f'unzip "{self._folder_path}/*.zip" -d {self._folder_path}')
        os.system(f"rm {self._folder_path}/*.zip")


    def train(self):
        print("All the code for training the model from scratch")


    def predict(self, source: str, target: str, options = {}):
        ''' All the code for getting predictions from the model. This function will be the main function called by the API and TRAPI queries
        Generates explanations based on the user input and returns a json
        '''
        # threshold_drugs : float,threshold_disease : float, features_drug, features_disease):
        evidence_path = self.generate_explanation(
            drug=source, disease=target,
            drug_fp_vectors = self.drug_fp_vectors,
            disease_hp_vectors= self.disease_hp_vectors,
            features_drug = options['features_drug'],
            features_disease = options['features_disease'],
            threshold_drugs = options['threshold_drugs'],
            threshold_disease = options['threshold_disease']
        )
        return self.generate_json(evidence_path)



    def __str__(self):
        json = {
            'folder_path': self._folder_path
        }
        return str(json)

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


    # df_op = pd.read_csv(f"{settings.OPENPREDICT_DATA_DIR}/resources/openpredict-omim-drug.csv")

    # drug_fp_vectors = KeyedVectors.load_word2vec_format(
    #     'openpredict/data/embedding/drugs_fp_embed.txt', binary=False)
    # disease_hp_vectors = KeyedVectors.load_word2vec_format(
    #     'openpredict/data/embedding/disease_hp_embed.txt', binary=False)

    # df_op = df_op.rename(columns={'omimid': 'disease_id', 'drugid': 'drug_id'})
    # df_op.disease_id = df_op.disease_id.astype(str)
    # (drug_ft_emb, disease_ft_emb) = load_treatment_embeddings('openpredict-baseline-omim-drugbank')

    # indications_dict = set()
    # for i, row in df_op.iterrows():
    #     #row['DB_ID'], row['DO_ID']
    #     pair = (str(row['drug_id']), str(row['disease_id']))
    #     indications_dict.add(pair)


    #functions which are used to generate the evidence path
    def generate_paths_for_apair(self, drug, disease, drug_emb_vectors, disease_emb_vectors,features_drug = None, features_disease = None,threshold_drugs = 1,threshold_diseases = 1):
        g = nx.Graph()
        (threshold_drug,threshold_disease) = self.getQuantiles(drug_emb_vectors, disease_emb_vectors, threshold_drugs)

        if(features_drug is not None) :
            filtered_embedding_drugs = KeyedVectors.load_word2vec_format(f'{settings.OPENPREDICT_DATA_DIR}/evidence-path-model/feature_{str(features_drug)}.txt', binary=False)
            similarDrugs = filtered_embedding_drugs.most_similar(drug, topn=100)
            (threshold_drug,threshold_disease) = self.getQuantiles(filtered_embedding_drugs, disease_emb_vectors, threshold_drugs)
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
                                                                "value": (1-sim)})
                g.add_node("OMIM:"+disease, id="OMIM:"+disease,
                name="fake", categories=["biolink:Disease"])
        (threshold_drug,threshold_disease) = self.getQuantiles(drug_emb_vectors, disease_emb_vectors, threshold_diseases)


        # TODO: USE settings.OPENPREDICT_DATA_DIR instead of lucky relative path
        if(features_disease is not None) :
            filtered_embedding_diseases = KeyedVectors.load_word2vec_format(f'{settings.OPENPREDICT_DATA_DIR}/evidence-path-model/feature_{str(features_disease)}.txt', binary=False)
            # filtered_embedding_diseases = KeyedVectors.load_word2vec_format(f'openpredict/data/embedding/feature_specific_embeddings_KG/feature_{str(features_disease)}.txt', binary=False)
            similarDiseases = filtered_embedding_diseases.most_similar(disease, topn=100)
            (threshold_drug,threshold_disease) = self.getQuantiles(self.drug_fp_vectors, filtered_embedding_diseases, threshold_diseases)
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

        for (dr, ds) in self.indications_dict:
            if "DRUGBANK:"+dr in g.nodes() and "OMIM:"+ds in g.nodes():
                g.add_edge("DRUGBANK:"+dr, "OMIM:"+ds, id="DRUGBANK:" +
                            dr+"_OMIM:"+ds, predicate="biolink:treats", subject="DRUGBANK:"+dr, object="OMIM:"+ds,  weight= 1.0,
                                                                                    attributes={"description": "score",
                                                                                "attribute_type_id": "EDAM:data_1772",
                                                                                "value": "1.0"
                                                                                })

        return (g)


    def generate_explanation(self, drug, disease, drug_fp_vectors, disease_hp_vectors,features_drug = None, features_disease = None,threshold_drugs = 0,threshold_disease = 0):
        #-> Path generation, add similar_to relation between query drug and disease
        #-> add known treats relations if any drug-disease pair in the graph has a treats relation
        g1= self.generate_paths_for_apair(
            drug, disease, drug_fp_vectors, disease_hp_vectors,features_drug, features_disease,threshold_drugs,threshold_disease)
        # Iterate over all simple paths
        # assign a weight to each path by summing their weights (for similar_to weight is 1-similarity, for treats, weight is 1)
        path_weight = self.path_weight_product(g1,drug=drug, disease=disease)
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


    def generate_json(self, graph) :
        graph_json ={}
        graph_json['nodes'] = list()

        for node in graph.nodes():
            graph_json['nodes'].append(graph[node])

        graph_json['edges']=list()
        for edge in graph.edges():
            graph_json['edges'].append(graph[edge[0]][edge[1]])

        return graph_json





    ###############################################
    # Script used to generate embeddings
    # to generate embeddings, call generate_feature_embedding_data() in method do_evidence_path()
    #
    def calculateEntitySimilarities(self, tokenized_vector, topn = 100) :
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



    def getQuantiles(self, drug_vectors, disease_vectors, quantile = 0.1) :
        ''' calulcates the nth quantile of the calculated similarity scores
            return : the min-threshold for the drugs and diseases as a tuple
        '''
        drug_similarities = self.calculateEntitySimilarities(drug_vectors,505)

        drug_sim_df = pd.DataFrame(drug_similarities)
        #print(drug_sim_df.describe())

        disease_similarities = self.calculateEntitySimilarities(disease_vectors,309)

        disease_sim_df = pd.DataFrame(disease_similarities)
        return ((drug_sim_df.quantile(quantile)[0]), (disease_sim_df.quantile(quantile)[0]))



    def percentiles_of_different_features(self):
        features_drug = ["TC", 'PPI_SIM', 'SE_SIM', 'GO_SIM', 'TARGETSEQ_SIM']
        features_diseases = ["HPO_SIM", 'PHENO_SIM']

        feature_percentiles = dict()
        for feature in features_drug :
            drug_emb = KeyedVectors.load_word2vec_format(
            'openpredict/data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDrugs.' + str(feature) + '.txt', binary=False)
            self.calculateEntitySimilarities(drug_emb)
            dr,ds = self.getQuantiles(drug_emb, self.disease_hp_vectors,1)
            feature_percentiles[feature] = dr

        for feature in features_diseases :
            disease_emb = KeyedVectors.load_word2vec_format(
            'openpredict/data/embedding/feature_specific_embeddings_KG/feature_FeatureTypesDiseases.' + str(feature) + '.txt', binary=False)
            self.calculateEntitySimilarities(disease_emb)
            dr,ds = self.getQuantiles(self.drug_fp_vectors, disease_emb,0.25)
            feature_percentiles[feature] = ds


        print(feature_percentiles)
        return feature_percentiles


    def path_weight_summation(self, g1,drug,disease):
        path_weight = {}
        for path in nx.all_simple_paths(g1,"DRUGBANK:"+drug,"OMIM:"+disease, cutoff=4):
            dpath = 0
            for i in range(len(path)-1):
                dpath += g1[path[i]][path[i+1]]['weight']
            path_weight[str(path)] = dpath

        return path_weight


    def path_weight_product(self, g1,drug,disease) :
        path_weight = {}
        for path in nx.all_simple_paths(g1,"DRUGBANK:"+drug,"OMIM:"+disease, cutoff=4):
            dpath = 0
            for i in range(len(path)-1):
                dpath *= g1[path[i]][path[i+1]]['weight']
            path_weight[str(path)] = dpath

        return path_weight



    def filter_out_features_diseases(self, features_of_interest):
        '''Creates the dataframe based on disease features to be converted to a embedding later '''

        resulting_embeddings = self.disease_ft_emb.loc[:,features_of_interest]
        #if(len(features_of_interest) > 1):
        #resulting_embeddings.columns = resulting_embeddings.columns.droplevel()
        self.save_embedding_as_txt(resulting_embeddings, str(features_of_interest) + ".txt")
        return resulting_embeddings

    def generate_feature_embedding_data(self):
        drug_features = {"GO_SIM", "PPI_SIM","SE_SIM","TARGETSEQ_SIM","TC"}
        disease_features = {"HPO_SIM", "PHENO_SIM"}

        for feature in drug_features :
            df = self.filter_out_features_drugs(feature)
            self.save_embedding_as_txt(df,'feature_FeatureTypesDrugs.' +feature+ '.txt')

        for feature in disease_features :
            df = self.filter_out_features_diseases(feature)
            self.save_embedding_as_txt(df,'feature_FeatureTypesDiseases.' +feature+ '.txt')

    def filter_out_features_drugs(self, features_of_interest) :
        '''Creates the dataframe based on drug features to be converted to a embedding later '''

        resulting_embeddings = self.drug_ft_emb.loc[:,features_of_interest]
        # if(len(features_of_interest) > 1) :
        #     resulting_embeddings.columns = resulting_embeddings.columns.droplevel()
        resulting_embeddings.index = [s.replace("DB", "") for s in list(resulting_embeddings.index.values)]
        self.save_embedding_as_txt(resulting_embeddings, str(features_of_interest) + ".txt")

        return resulting_embeddings


    def save_embedding_as_txt(self, embedding_df, fileName) :
        '''
        takes the dataframe filtered based on the features and returns a txt which represents
        its embedding
        '''
        embedding_df.index = list(map(int, embedding_df.index))
        embedding_df = embedding_df.reset_index()
        embedding_df_np = embedding_df.to_numpy()
        np.savetxt('openpredict/data/embedding/feature_' + fileName, embedding_df_np, fmt = '%f' )
