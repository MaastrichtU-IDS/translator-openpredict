# @Author Arif YILMAZ, a.yilmaz@maastrichtuniversity.nl
# @repoaddress "https://github.com/arifx/XPREDICT"

import csv

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as fn

from drkg_model.download import download
from openpredict.utils import get_entities_labels, get_openpredict_dir

# Predict drug repurposing based on the DRKG (drug repurposing KG) by Arif Yilmaz

def predictDrugRepositioning(diseaseCURIElist,noofResults):
    # FILEPATH="/openpredict/data/kgpredict/"
    FILEPATH=get_openpredict_dir("kgpredict/")
    EMBPATH=FILEPATH+"embed/"

    try:

        diseaselist= [diseaseCURIElist]
        #print("DISLIST:"+str(diseaselist))
        drug_list = []

        with open(FILEPATH+"kgpredict_drug_diseasemappings.tsv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
            for row_val in reader:
                drug_list.append(row_val['drug'])

        len(drug_list)

        treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']

        entity_idmap_file = EMBPATH+'entities.tsv'
        relation_idmap_file = EMBPATH+'relations.tsv'

        entity_map = {}
        entity_id_map = {}
        relation_map = {}

        with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:

            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
            for row_val in reader:
                entity_map[row_val['name']] = int(row_val['id'])
                entity_id_map[int(row_val['id'])] = row_val['name']

        with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
            for row_val in reader:
                relation_map[row_val['name']] = int(row_val['id'])

        drug_ids = []
        disease_ids = []
        for drug in drug_list:
            drug_ids.append(entity_map[drug])

        for disease in diseaselist:
            disease_ids.append(entity_map["Disease::"+str(disease)])

        treatment_rid = [relation_map[treat]  for treat in treatment]

        # Load embeddings
        pth=EMBPATH+'DRKG_TransE_l2_entity.npy'
        #print("DISLIST3:"+str(pth))
        entity_emb = np.load(pth)
        #np.savez_compressed("DRKG_TransE_l2_entity.npz", entity_emb=entity_emb)
        rel_emb = np.load(EMBPATH+'DRKG_TransE_l2_relation.npy')

        drug_ids = th.tensor(drug_ids).long()
        disease_ids = th.tensor(disease_ids).long()
        treatment_rid = th.tensor(treatment_rid)

        drug_emb = th.tensor(entity_emb[drug_ids])
        treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]

        gamma=12.0
        def getEmbedding(head, rel, tail):
            score = head + rel - tail
            return gamma - th.norm(score, p=2, dim=-1)

        scores_per_disease = []
        dids = []
        for rid in range(len(treatment_embs)):
            treatment_emb=treatment_embs[rid]
            for disease_id in disease_ids:
                disease_emb = entity_emb[disease_id]
                score = fn.logsigmoid(getEmbedding(drug_emb, treatment_emb, disease_emb))
                scores_per_disease.append(score)
                dids.append(drug_ids)
        scores = th.cat(scores_per_disease)
        dids = th.cat(dids)

        # sort scores
        idx = th.flip(th.argsort(scores), dims=[0])
        scores = scores[idx].numpy()
        dids = dids[idx].numpy()


        _, unique_indices = np.unique(dids, return_index=True)
        topk=noofResults
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_dids = dids[topk_indices]
        proposed_scores = scores[topk_indices]


        for i in range(topk):
            drug = int(proposed_dids[i])
            score = proposed_scores[i]

        proposed_drugnames={}

        for row_val in range(topk):
            drug = int(proposed_dids[row_val])
            proposed_drugnames[row_val] = entity_id_map[drug].replace("Compound::","")


        prediction_df = pd.DataFrame(list(zip(
            proposed_drugnames.values(), proposed_drugnames.values(), list(proposed_scores))), columns=['drug', 'disease', 'score'])
        prediction_df.sort_values(by='score', inplace=True, ascending=False)
        # prediction_df = pd.DataFrame( list(zip(pairs[:,0], pairs[:,1], y_proba[:,1])), columns =[drug_column_label,disease_column_label,'score'])

        # Add namespace to get CURIEs from IDs
        prediction_df["drug"] = "DRUGBANK:" + prediction_df["drug"]
        prediction_df["disease"] =  prediction_df["disease"]

        # prediction_results=prediction_df.to_json(orient='records')
        prediction_results = prediction_df.to_dict(orient='records')
        return prediction_results
    except Exception as e:
        print(f"Error getting predictions for DRKG model: {e}")
        return pd.DataFrame()




#print("Drug Repositioning  TOP 100 Predictions (Score=0 is best drug) :")
#drugscores=predictDrugRepositioning(uterinecancerdisease,noofResults=100)
#print(drugscores)




def get_drugrepositioning_results(
        diseaseCURIElist, n_results=100,
    ):
    """Run kg evaluation to get predictions

    :param score: score minimum of predictions
    :param n_results: number of predictions to return
    :return: predictions in array of JSON object
    """
    download()
    # classifier: Predict OMIM-DrugBank
    # TODO: improve when we will have more classifier
    id_to_predict=diseaseCURIElist
    predictions_array = predictDrugRepositioning(diseaseCURIElist,n_results)

    if n_results:
        # Predictions are already sorted from higher score to lower
        predictions_df = predictions_array[:n_results]
    predictions_df
    # Build lists of unique node IDs to retrieve label
    predicted_ids = set()
    for prediction in predictions_array:
        for key, value in prediction.items():
            if key != 'score':
                predicted_ids.add(value)
    labels_dict = get_entities_labels(predicted_ids)

    # TODO: format using a model similar to BioThings:
    # cf. at the end of this file

    # Add label for each ID, and reformat the dict using source/target
    labelled_predictions = []
    # Second array with source and target info for the reasoner query resolution
    for prediction in predictions_array:
        labelled_prediction = {}
        for key, value in prediction.items():
            if key == 'score':
                labelled_prediction['score'] = value
            elif value != id_to_predict:
                labelled_prediction['id'] = value
                labelled_prediction['type'] = key
                #print("SHAPX:"+value)
                #SHAPDISABLE shaps=xp.getXPREDICTExplanation(drugId=value)

                #SHAPDISABLE labelled_prediction['shap'] = shaps
                # Same for source_target object
                try:
                    if value in labels_dict and labels_dict[value]:
                        labelled_prediction['label'] = labels_dict[value]['id']['label']
                except:
                    print('No label found for ' + value)
                # if value in labels_dict and labels_dict[value] and labels_dict[value]['id'] and labels_dict[value]['id']['label']:
                #     labelled_prediction['label'] = labels_dict[value]['id']['label']
                #     source_target_prediction['target']['label'] = labels_dict[value]['id']['label']

        labelled_predictions.append(labelled_prediction)

    return labelled_predictions
