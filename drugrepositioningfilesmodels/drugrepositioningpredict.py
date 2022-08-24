# @Author Arif YILMAZ, a.yilmaz@maastrichtuniversity.nl
#@repoaddress "https://github.com/arifx/XPREDICT"

import csv

import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as fn
import pdb
import os
import sys


alz_diseaselist= ['Disease::MESH:D000544']
alz_diseaselist2=['Disease::MESH:C566298']
uterinecancerdisease=['Disease::DOID:363']
diseaselist=uterinecancerdisease

def predictDrugRepositioning(diseaseCURIElist,noofResults):
  try:
    #cwd = os.getcwd() + "/translator-openpredict/drugrepositioningfilesmodels"
    cwd = os.getcwd() + "/drugrepositioningfilesmodels"
    #pdb.set_trace()
    diseaselist= [diseaseCURIElist]
    #print("DISLIST:"+str(diseaselist))
    drug_list = []
    with open(cwd+"/infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])
    
    len(drug_list)
        
    treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']
    
    entity_idmap_file = cwd+'/entities.tsv'
    relation_idmap_file = cwd+'/relations.tsv'
    
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
    #pdb.set_trace()
    # Load embeddings
    pth=cwd+'/entity_embeddings.npy'
    #print("DISLIST3:"+str(pth))  
    entity_emb = np.load(pth)
    #np.savez_compressed("entity_embeddings.npz", entity_emb=entity_emb)
    rel_emb = np.load(cwd+'/relation_embeddings.npy')

         
    drug_ids = th.tensor(drug_ids).long()
    disease_ids = th.tensor(disease_ids).long()
    treatment_rid = th.tensor(treatment_rid)
    
    drug_emb = th.tensor(entity_emb[drug_ids])
    treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]
#pdb.set_trace()
    
    gamma=12.0
    def transE_l2(head, rel, tail):
        score = head + rel - tail
        return gamma - th.norm(score, p=2, dim=-1)
   
    scores_per_disease = []
    dids = []
    for rid in range(len(treatment_embs)):
        treatment_emb=treatment_embs[rid]
        for disease_id in disease_ids:
            disease_emb = entity_emb[disease_id]
            score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
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
    
  except Exception as e:
     print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
     
  return prediction_results




   
   
#print("Drug Repositioning  TOP 100 Predictions (Score=0 is best drug) :")
#drugscores=predictDrugRepositioning(uterinecancerdisease,noofResults=100)    
#print(drugscores)
    
    
     

