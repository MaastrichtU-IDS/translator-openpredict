import os
import re
from typing import List

import numpy as np
import pandas as pd

from openpredict.config import settings
from openpredict.loaded_models import PreloadedModels
from openpredict.models.predict_output import PredictOutput, TrapiRelation
from openpredict.utils import get_entities_labels
from openpredict_model.train import createFeaturesSparkOrDF

satisfies_relations: List[TrapiRelation] = [
    {
        'subject': 'biolink:Drug',
        'predicate': 'biolink:treats',
        'object': 'biolink:Disease',
    },
    {
        'subject': 'biolink:Disease',
        'predicate': 'biolink:treated_by',
        'object': 'biolink:Drug',
    },
]

def get_predictions(
        id_to_predict, model_id, min_score=None, max_score=None, n_results=None,
    ) -> PredictOutput:
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
                try:
                    if value in labels_dict and labels_dict[value]:
                        labelled_prediction['label'] = labels_dict[value]['id']['label']
                except:
                    print('No label found for ' + value)
                # if value in labels_dict and labels_dict[value] and labels_dict[value]['id'] and labels_dict[value]['id']['label']:
                #     labelled_prediction['label'] = labels_dict[value]['id']['label']

        labelled_predictions.append(labelled_prediction)

    return labelled_predictions



def query_omim_drugbank_classifier(input_curie, model_id):
    """The main function to query the drug-disease OpenPredict classifier,
    It queries the previously generated classifier a `.joblib` file
    in the `data/models` folder

    :return: Predictions and scores
    """
    # TODO: XAI add the additional scores from SHAP here

    parsed_curie = re.search('(.*?):(.*)', input_curie)
    input_namespace = parsed_curie.group(1)
    input_id = parsed_curie.group(2)

    # resources_folder = "data/resources/"
    # features_folder = "data/features/"
    # drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv',
    #                 'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    # diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    # drugfeatfiles = [ os.path.join(features_folder, fn) for fn in drugfeatfiles]
    # diseasefeatfiles = [ os.path.join(features_folder, fn) for fn in diseasefeatfiles]

    # Get all DFs
    # Merge feature matrix
    # drug_df, disease_df = mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
    # (drug_df, disease_df)= load('data/features/drug_disease_dataframes.joblib')

    (drug_df, disease_df) = PreloadedModels.treatment_embeddings

    # TODO: should we update this file too when we create new runs?
    drugDiseaseKnown = pd.read_csv(
        os.path.join(settings.OPENPREDICT_DATA_DIR, 'resources', 'openpredict-omim-drug.csv'), delimiter=',')
    drugDiseaseKnown.rename(
        columns={'drugid': 'Drug', 'omimid': 'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)
    print('Known indications', len(drugDiseaseKnown))

    # TODO: save json?
    drugDiseaseDict = {
        tuple(x) for x in drugDiseaseKnown[['Drug', 'Disease']].values}

    drugwithfeatures = set(drug_df.columns.levels[1].tolist())
    diseaseswithfeatures = set(disease_df.columns.levels[1].tolist())

    # TODO: save json?
    commonDrugs = drugwithfeatures.intersection(drugDiseaseKnown.Drug.unique())
    commonDiseases = diseaseswithfeatures.intersection(
        drugDiseaseKnown.Disease.unique())

    clf = PreloadedModels.treatment_classifier
    # if not clf:
    #     clf = load_treatment_classifier(model_id)

    # print("📥 Loading classifier " +
    #       get_openpredict_dir('models/' + model_id + '.joblib'))
    # clf = load(get_openpredict_dir('models/' + model_id + '.joblib'))

    pairs = []
    classes = []
    if input_namespace.lower() == "drugbank":
        # Input is a drug, we only iterate on disease
        dr = input_id
        # drug_column_label = "source"
        # disease_column_label = "target"
        for di in commonDiseases:
            cls = (1 if (dr, di) in drugDiseaseDict else 0)
            pairs.append((dr, di))
            classes.append(cls)
    else:
        # Input is a disease
        di = input_id
        # drug_column_label = "target"
        # disease_column_label = "source"
        for dr in commonDrugs:
            cls = (1 if (dr, di) in drugDiseaseDict else 0)
            pairs.append((dr, di))
            classes.append(cls)

    classes = np.array(classes)
    pairs = np.array(pairs)

    test_df = createFeaturesSparkOrDF(pairs, classes, drug_df, disease_df)

    # Get list of drug-disease pairs (should be saved somewhere from previous computer?)
    # Another API: given the type, what kind of entities exists?
    # Getting list of Drugs and Diseases:
    # commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    # commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )
    features = list(test_df.columns.difference(['Drug', 'Disease', 'Class']))
    y_proba = clf.predict_proba(test_df[features])

    prediction_df = pd.DataFrame(list(zip(
        pairs[:, 0], pairs[:, 1], y_proba[:, 1])), columns=['drug', 'disease', 'score'])
    prediction_df.sort_values(by='score', inplace=True, ascending=False)
    # prediction_df = pd.DataFrame( list(zip(pairs[:,0], pairs[:,1], y_proba[:,1])), columns =[drug_column_label,disease_column_label,'score'])

    # Add namespace to get CURIEs from IDs
    prediction_df["drug"] = "DRUGBANK:" + prediction_df["drug"]
    prediction_df["disease"] = "OMIM:" + prediction_df["disease"]

    # prediction_results=prediction_df.to_json(orient='records')
    prediction_results = prediction_df.to_dict(orient='records')
    return prediction_results


def get_similar_for_entity(input_curie, emb_vectors, n_results):
    print (input_curie)
    parsed_curie = re.search('(.*?):(.*)', input_curie)
    input_namespace = parsed_curie.group(1)
    input_id = parsed_curie.group(2)

    drug = None
    disease = None
    if input_namespace.lower() == "drugbank":
        drug = input_id
    else:
        disease = input_id

    #g= Graph()
    if n_results == None:
        n_results = len(emb_vectors.vocab)

    similar_entites = []
    if  drug is not None and drug in emb_vectors:
        similarDrugs = emb_vectors.most_similar(drug,topn=n_results)
        for en,sim in similarDrugs:
            #g.add((DRUGB[dr],BIOLINK['treats'],OMIM[ds]))
            #g.add((DRUGB[dr], BIOLINK['similar_to'],DRUGB[drug]))
            similar_entites.append((drug,en,sim))
    if  disease is not None and disease in emb_vectors:
        similarDiseases = emb_vectors.most_similar(disease,topn=n_results)
        for en,sim in similarDiseases:
            similar_entites.append((disease, en, sim))



    if drug is not None:
        similarity_df = pd.DataFrame(similar_entites, columns=['entity', 'drug', 'score'])
        similarity_df["entity"] = "DRUGBANK:" + similarity_df["entity"]
        similarity_df["drug"] = "DRUGBANK:" + similarity_df["drug"]

    if disease is not None:
        similarity_df = pd.DataFrame(similar_entites, columns=['entity', 'disease', 'score'])
        similarity_df["entity"] = "OMIM:" + similarity_df["entity"]
        similarity_df["disease"] = "OMIM:" + similarity_df["disease"]

    # prediction_results=prediction_df.to_json(orient='records')
    similarity_results = similarity_df.to_dict(orient='records')
    return similarity_results



def get_similarities(types, id_to_predict, emb_vectors, min_score=None, max_score=None, n_results=None):
    """Run classifiers to get predictions

    :param id_to_predict: Id of the entity to get prediction from
    :param classifier: classifier used to get the predictions
    :param score: score minimum of predictions
    :param n_results: number of predictions to return
    :return: predictions in array of JSON object
    """
    predictions_array = get_similar_for_entity(id_to_predict, emb_vectors, n_results)

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

    labelled_predictions = []
    for prediction in predictions_array:
        labelled_prediction = {}
        for key, value in prediction.items():
            if key == 'score':
                labelled_prediction['score'] = value
            elif value != id_to_predict:
                labelled_prediction['id'] = value
                labelled_prediction['type'] = key
                try:
                    if value in labels_dict and labels_dict[value]:
                        labelled_prediction['label'] = labels_dict[value]['id']['label']
                except:
                    print('No label found for ' + value)
                # if value in labels_dict and labels_dict[value] and labels_dict[value]['id'] and labels_dict[value]['id']['label']:
                #     labelled_prediction['label'] = labels_dict[value]['id']['label']

        labelled_predictions.append(labelled_prediction)

    return labelled_predictions
