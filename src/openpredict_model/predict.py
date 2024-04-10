import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from trapi_predict_kit import load, PredictInput, PredictOutput, trapi_predict, get_entities_labels, get_entity_types, log
from openpredict_model.utils import load_features_embeddings, load_similarity_embeddings, get_openpredict_dir, resolve_ids_with_nodenormalization_api, createFeaturesSparkOrDF

trapi_nodes = {
    "biolink:Disease": {
        "id_prefixes": [
            "OMIM"
        ]
    },
    "biolink:Drug": {
        "id_prefixes": [
            "DRUGBANK"
        ]
    }
}


@trapi_predict(
    path='/predict',
    name="Get predicted targets for given entities",
    description="""Return the predicted targets for given entities: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.
Provide the list of drugs as `subjects`, and/or the list of diseases as `objects`
This operation is annotated with x-bte-kgs-operations, and follow the BioThings API recommendations.

You can try:

| objects: `OMIM:246300` | subjects: `DRUGBANK:DB00394` |
| ------- | ---- |
| to check the drug predictions for a disease   | to check the disease predictions for a drug |
""",
    edges=[
        {
            'subject': 'biolink:Drug',
            'predicate': 'biolink:treats',
            'inverse': 'biolink:treated_by',
            'object': 'biolink:Disease',
            'relations': [
                'RO:0002434'
            ],
        },
    ],
    nodes=trapi_nodes
)
def get_predictions(request: PredictInput) -> PredictOutput:
    """Run classifiers to get predictions

    :param input_id: Id of the entity to get prediction from
    :param classifier: classifier used to get the predictions
    :param score: score minimum of predictions
    :param n_results: number of predictions to return
    :return: predictions in array of JSON object
    """
    if request.options.model_id is None:
        request.options.model_id = 'openpredict_baseline'

    subjects = request.subjects
    if not subjects:
        subjects = request.objects

    trapi_to_supported, supported_to_trapi = resolve_ids_with_nodenormalization_api(
        request.subjects + request.objects
    )

    labelled_predictions = []
    for subject in subjects:
        supported_subject = trapi_to_supported.get(subject, subject)

        # classifier: Predict OMIM-DrugBank
        # TODO: improve when we will have more classifier
        predictions_array = query_omim_drugbank_classifier(supported_subject, request.options.model_id)

        if request.options.min_score:
            predictions_array = [
                p for p in predictions_array if p['score'] >= request.options.min_score]
        if request.options.max_score:
            predictions_array = [
                p for p in predictions_array if p['score'] <= request.options.max_score]
        if request.options.n_results:
            # Predictions are already sorted from higher score to lower
            predictions_array = predictions_array[:request.options.n_results]

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
        # Second array with source and target info for the reasoner query resolution
        for prediction in predictions_array:
            trapi_subject = supported_to_trapi.get(prediction["drug"], prediction["drug"])
            trapi_object = supported_to_trapi.get(prediction["disease"], prediction["disease"])
            # log.info(prediction)
            if request.subjects and trapi_subject not in request.subjects:
                continue
            if request.objects and trapi_object not in request.objects:
                continue  # Don't add the prediction if not in the list of requested objects
            labelled_prediction = {
                "subject": trapi_subject,
                "object": trapi_object,
                "score": prediction["score"],
            }
            if "drug" in prediction and prediction["drug"] in labels_dict and labels_dict[prediction["drug"]]:
                labelled_prediction['subject_label'] = labels_dict[prediction["drug"]]['id']['label']
            if "disease" in prediction and prediction["disease"] in labels_dict and labels_dict[prediction["disease"]]:
                labelled_prediction['object_label'] = labels_dict[prediction["disease"]]['id']['label']
            labelled_predictions.append(labelled_prediction)
    log.info(labelled_predictions)
    return {'hits': labelled_predictions, 'count': len(labelled_predictions)}



def query_omim_drugbank_classifier(input_curie, model_id):
    """The main function to query the drug-disease OpenPredict classifier,
    It queries the previously generated classifier a `.pickle` file
    in the `data/models` folder

    :return: Predictions and scores
    """
    # TODO: XAI add the additional scores from SHAP here
    loaded_model = load(f"models/{model_id}")
    clf  = loaded_model.model
    (drug_df, disease_df)  = load_features_embeddings(model_id)

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
    # (drug_df, disease_df)= load('data/features/drug_disease_dataframes.pickle')

    # (drug_df, disease_df) = load_treatment_embeddings(model_id)

    # TODO: should we update this file too when we create new runs?
    drugDiseaseKnown = pd.read_csv(
        os.path.join(get_openpredict_dir(), 'resources', 'openpredict-omim-drug.csv'), delimiter=',')
    drugDiseaseKnown.rename(
        columns={'drugid': 'Drug', 'omimid': 'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)
    log.debug('Known indications', len(drugDiseaseKnown))

    # TODO: save json?
    drugDiseaseDict = {
        tuple(x) for x in drugDiseaseKnown[['Drug', 'Disease']].values}

    drugwithfeatures = set(drug_df.columns.levels[1].tolist())
    diseaseswithfeatures = set(disease_df.columns.levels[1].tolist())

    # TODO: test with OMIM:618077
    if input_id not in drugwithfeatures.union(diseaseswithfeatures):
        log.warning(f"No features for {input_curie}")
        return []

    # TODO: save json?
    commonDrugs = drugwithfeatures.intersection(drugDiseaseKnown.Drug.unique())
    commonDiseases = diseaseswithfeatures.intersection(
        drugDiseaseKnown.Disease.unique())

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


def get_similar_for_entity(input_curie: str, emb_vectors, n_results: int = None):
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
    if not n_results:
        n_results = len(emb_vectors.vocab)

    similar_entites = []
    if drug is not None and drug in emb_vectors:
        similarDrugs = emb_vectors.most_similar(drug,topn=n_results)
        for en,sim in similarDrugs:
            #g.add((DRUGB[dr],BIOLINK['treats'],OMIM[ds]))
            #g.add((DRUGB[dr], BIOLINK['similar_to'],DRUGB[drug]))
            similar_entites.append((drug,en,sim))
    if disease is not None and disease in emb_vectors:
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



@trapi_predict(
    path='/similarity',
    name="Get similar entities",
    default_input="DRUGBANK:DB00394",
    default_model=None,
    edges=[
        {
            'subject': 'biolink:Drug',
            'predicate': 'biolink:similar_to',
            'object': 'biolink:Drug',
        },
        {
            'subject': 'biolink:Disease',
            'predicate': 'biolink:similar_to',
            'object': 'biolink:Disease',
        },
    ],
    nodes=trapi_nodes,
    description="""Get similar entites for a given entity CURIE.

You can try:

| drug_id: `DRUGBANK:DB00394` | disease_id: `OMIM:246300` |
| ------- | ---- |
| model_id: `drugs_fp_embed.txt` | model_id: `disease_hp_embed.txt` |
| to check the drugs similar to a given drug | to check the diseases similar to a given disease   |
""",
)
def get_similarities(request: PredictInput):
    """Run classifiers to get predictions

    :param input_id: Id of the entity to get prediction from
    :param options
    :return: predictions in array of JSON object
    """
    labelled_predictions = []
    trapi_to_supported, supported_to_trapi = resolve_ids_with_nodenormalization_api(
        request.subjects + request.objects
    )

    for subject in request.subjects:
        if not request.options.model_id:
            request.options.model_id = 'drugs_fp_embed.txt'
            input_types = get_entity_types(subject)
            if 'biolink:Disease' in input_types:
                request.options.model_id = 'disease_hp_embed.txt'
            # if len(input_types) == 0:
            #     # If no type found we try to check from the ID namespace
            #     if input_id.lower().startswith('omim:'):
            #         options.model_id = 'disease_hp_embed.txt'
        emb_vectors = load_similarity_embeddings(request.options.model_id)

        supported_subject = trapi_to_supported.get(subject, subject)
        predictions_array = get_similar_for_entity(supported_subject, emb_vectors, request.options.n_results)

        if request.options.min_score:
            predictions_array = [
                p for p in predictions_array if p['score'] >= request.options.min_score]
        if request.options.max_score:
            predictions_array = [
                p for p in predictions_array if p['score'] <= request.options.max_score]
        if request.options.n_results:
            # Predictions are already sorted from higher score to lower
            predictions_array = predictions_array[:request.options.n_results]

        # Build lists of unique node IDs to retrieve label
        predicted_ids = set()
        for prediction in predictions_array:
            for key, value in prediction.items():
                if key != 'score':
                    predicted_ids.add(value)
        labels_dict = get_entities_labels(predicted_ids)

        for prediction in predictions_array:
            trapi_subject = supported_to_trapi.get(prediction["entity"], prediction["entity"])
            object_id = prediction["disease"] if "disease" in prediction else prediction["drug"]
            trapi_object = supported_to_trapi.get(object_id, object_id)

            labelled_prediction = {"subject": subject}
            labelled_prediction = {
                "subject": trapi_subject,
                "object": trapi_object,
                "score": prediction["score"],
            }
            if "drug" in prediction and prediction["drug"] in labels_dict and labels_dict[prediction["drug"]]:
                labelled_prediction['subject_label'] = labels_dict[prediction["drug"]]['id']['label']
            if "disease" in prediction and prediction["disease"] in labels_dict and labels_dict[prediction["disease"]]:
                labelled_prediction['object_label'] = labels_dict[prediction["disease"]]['id']['label']

            if request.subjects and labelled_prediction["subject"] not in request.subjects:
                continue
            if request.objects and labelled_predictions["object"] not in request.objects:
                continue  # Don't add the prediction if not in the list of requested objects
            labelled_predictions.append(labelled_prediction)

    return {'hits': labelled_predictions, 'count': len(labelled_predictions)}
