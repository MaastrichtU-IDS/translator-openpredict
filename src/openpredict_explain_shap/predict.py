# @Author Arif YILMAZ, a.yilmaz@maastrichtuniversity.nl
# @repoaddress "https://github.com/arifx/XPREDICT"


import pandas as pd
import shap as shap
import sklearn

from openpredict.utils import get_entities_labels, get_openpredict_dir
from openpredict_model.predict import query_omim_drugbank_classifier

# XPREDICT framework may be used to add explanation features to drug repositioning applications

def getSHAPModel(datasetFile="xpredict/deepdrug_repurposingpredictiondataset.csv", mlmodel="LR"):
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

    deepdrug_dataset_df=pd.read_csv(get_openpredict_dir(datasetFile))
    Xdf= deepdrug_dataset_df[features]
    ydf=deepdrug_dataset_df[outclass]

    modellr = sklearn.linear_model.LogisticRegression()
    modellr.fit(Xdf, ydf)


    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    X100 = shap.utils.sample(Xdf, 50130)
    explainerlr = shap.Explainer(modellr,X100)

    shap_valueslr = explainerlr(Xdf)

    return shap_valueslr


def getXPREDICTExplanation(shap_values,drugId="000"):

    if drugId=="000":
        return pd.DataFrame().to_json()

    ##shap.plots.bar(shap_valueslr,max_display=11)
    ##shap.plots.beeswarm(shap_valueslr)
    # shap_valueslr[1]
    # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_valueslr[1])

    datasetFile="xpredict/deepdrug_repurposingpredictiondataset.csv"
    deepdrug_dataset_df=pd.read_csv(get_openpredict_dir(datasetFile))

    i=50#deepdrug_dataset_df['Drug'== drugId]
    strx=str(drugId)
    drid=strx[strx.rfind(":")+1:]
    #print("DRID:"+ str( drid))
    #print("DURGLIST:"+str(deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==drid]))
    if deepdrug_dataset_df.shape[0] >0:
        #shapx=deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==drid].iloc[0] #test
        ishap=deepdrug_dataset_df.loc[deepdrug_dataset_df.Drug==drid].index[0]

        shapx=shap_values[ishap]
        #print("ISHAP:"+str("Shapley Values:"+str(shapx)))
        # print(f"shapex type: {type(shapx)}")

        shapx_list = []
        feature_list = [
            "TC_HPO-SIM", "TC_PHENO-SIM", "SE-SIM_HPO-SIM", "SE-SIM_PHENO-SIM", "PPI-SIM_HPO-SIM", "PPI-SIM_PHENO-SIM",
            "HPO-SIM", "PHENO-SIM", "GO-SIM_HPO-SIM", "GO-SIM_PHENO-SIM", "TARGETSEQ-SIM_HPO-SIM", "TARGETSEQ-SIM_PHENO-SIM"
        ]
        i = 0
        for sh in shapx:
            shapx_list.append({
                'feature': feature_list[i],
                'values': sh.values,
                'base_values': sh.base_values,
                'data': sh.data,
            })
            i += 1

        # return str(shapx)
        return shapx_list

    else:
        print("test ravel3")
        return pd.DataFrame().to_json() #return null DF




def get_explanations(
        id_to_predict, model_id, app,
        min_score=None, max_score=None, n_results=None,
    ):
    """Run classifiers to get predictions

    :param id_to_predict: Id of the entity to get prediction from
    :param classifier: classifier used to get the predictions
    :param score: score minimum of predictions
    :param n_results: number of predictions to return
    :return: predictions in array of JSON object
    """
    # classifier: Predict OMIM-DrugBank
    # TODO: improve when we will have more classifier
    predictions_array = query_omim_drugbank_classifier(id_to_predict, model_id)
    shap_valueslr= getSHAPModel(datasetFile="xpredict/deepdrug_repurposingpredictiondataset.csv")

    if min_score:
        predictions_array = [
            p for p in predictions_array if p['score'] >= min_score]
    if max_score:
        predictions_array = [
            p for p in predictions_array if p['score'] <= max_score]
    if n_results:
        # Predictions are already sorted from higher score to lower
        predictions_array = predictions_array[:n_results]

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

                shaps=getXPREDICTExplanation(shap_values=shap_valueslr, drugId=value)

                labelled_prediction['shap'] = shaps
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
