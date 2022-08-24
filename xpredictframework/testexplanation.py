
import xpredict as xp
import shap
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from subprocess import check_output
import shap as shap
import os
import sklearn

import pandas as pd
drugId="DB00915"
shapid="104300"
cwd = os.getcwd() +"/xpredictframework/"



def getXPREDICTExplanation(datasetFile="deepdrug_repurposingpredictiondataset.csv",mlmodel="LR",drugId="000"):
    cwd = os.getcwd()+"/xpredictframework/"
    if drugId==0 :
        return "[{}]"
    
    features= ['GO-SIM_HPO-SIM',
     'GO-SIM_PHENO-SIM',
     'PPI-SIM_HPO-SIM',
     'PPI-SIM_PHENO-SIM',
     'SE-SIM_HPO-SIM',
     'SE-SIM_PHENO-SIM',
     'TARGETSEQ-SIM_HPO-SIM',
     'TARGETSEQ-SIM_PHENO-SIM',
     'TC_HPO-SIM',
     'TC_PHENO-SIM']
    outclass=['Class']
    

    deepdrug_dataset_df=pd.read_csv(cwd+datasetFile)
    Xdf= deepdrug_dataset_df[features]
    ydf=deepdrug_dataset_df[outclass]
    modellr = sklearn.linear_model.LogisticRegression()
    modellr.fit(Xdf, ydf)
    
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    X100 = shap.utils.sample(Xdf, 50130)
    explainerlr = shap.Explainer(modellr,X100)
    shap_valueslr = explainerlr(Xdf)
    
    
    ##shap.plots.bar(shap_valueslr,max_display=11)
    ##shap.plots.beeswarm(shap_valueslr)
    # shap_valueslr[1]
    # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_valueslr[1])
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #alz = [55, 534, 697, 1039, 1182]
    #0915 = 0,1,2
    #0268 =173,174,175,176
    #00715 = 4548,4549,4550
    #00734= 536, 537,538,539,540,541,542
    # 1224= 1378,1379
    #1238 = 552,553
    
    alz = [1,176,4548, 542,1379,553]

    #plt.show()
    
    i=50#deepdrug_dataset_df['Drug'== drugId]
    shapid="104300"
    value="DB00915"
    #druglist=deepdrug_dataset_df.loc[Drug==shapid]
    #print(str(druglist))
    #print("explainable parts2:")
    strx=str(value)
    drid=strx[strx.rfind(":")+1:]
    print("DRID:"+ str( drid))
    shaps=str(xp.getXPREDICTExplanation(drugId=drid))
    shapx=deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==drugId].iloc[0]
    shapjson=shapx.to_json()
    #print("SHAXX:"+shapx)
    return shapjson



getXPREDICTExplanation(drugId="DB00915")




features= ['GO-SIM_HPO-SIM',
    'GO-SIM_PHENO-SIM',
    'PPI-SIM_HPO-SIM',
    'PPI-SIM_PHENO-SIM',
    'SE-SIM_HPO-SIM',
    'SE-SIM_PHENO-SIM',
    'TARGETSEQ-SIM_HPO-SIM',
    'TARGETSEQ-SIM_PHENO-SIM',
    'TC_HPO-SIM',
    'TC_PHENO-SIM']
outclass=['Class']
datasetFile="deepdrug_repurposingpredictiondataset.csv"
deepdrug_dataset_df=pd.read_csv(cwd+datasetFile)

#druglist=deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==shapid].iloc[0]


shapx=deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==drugId].iloc[0]
#shapvs=xp.getXPREDICTExplanation(drugId=drid)

print(shapvs)

