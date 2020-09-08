import logging
import requests
from openpredict.openpredict_omim_drugbank import query_omim_drugbank_classifier

def get_predictions(id_to_predict, classifier='OpenPredict OMIM-DrugBank', score=None, limit=None):
    """Run classifiers to get predictions

    :param id_to_predict: Id of the entity to get prediction from
    :param classifier: classifier used to get the predictions
    :param score: score minimum of predictions
    :param limit: limit number of predictions to return
    :return: predictions in array of JSON object
    """
    # TODO: improve when we will have more classifier
    predictions_array = query_omim_drugbank_classifier(id_to_predict)
    
    if score:
        predictions_array = [p for p in predictions_array if p['score'] >= score]
    if limit:
        # Predictions are already sorted from higher score to lower
        predictions_array = predictions_array[:limit]
    
    # Build lists of unique node IDs to retrieve label
    predicted_ids = set([])
    for prediction in predictions_array:
        for key, value in prediction.items():
            if key != 'score':
                predicted_ids.add(value)
    labels_dict = get_labels(predicted_ids)

    # Add label for each ID, and reformat the dict using source/target
    labelled_predictions = []
    for prediction in predictions_array:
        labelled_prediction = {}
        for key, value in prediction.items():
            if key == 'score':
                labelled_prediction['score'] = value
            elif value == id_to_predict:
                labelled_prediction['source'] = {
                    'id': id_to_predict,
                    'type': key
                }
                if id_to_predict in labels_dict and labels_dict[id_to_predict]:
                    labelled_prediction['source']['label'] = labels_dict[id_to_predict]['id']['label']
            else:
                # Then it is the target node
                labelled_prediction['target'] = {
                    'id': value,
                    'type': key
                }
                if value in labels_dict and labels_dict[value]:
                    labelled_prediction['target']['label'] = labels_dict[value]['id']['label']
        labelled_predictions.append(labelled_prediction)
        # returns
        # { score: 12,
        #  source: {
        #      id: DB0001
        #      type: drug,
        #      label: a drug
        #  },
        #  target { .... }}
    
    return labelled_predictions

def get_labels(entity_list):
    """Send the list of node IDs to Translator Normalization API to get labels
    See API: https://nodenormalization-sri.renci.org/apidocs/#/Interfaces/get_get_normalized_nodes
    and example notebook: https://github.com/TranslatorIIPrototypes/NodeNormalization/blob/master/documentation/NodeNormalization.ipynb
    """
    # TODO: add the preferred identifier CURIE to our answer also?
    try:
        get_label_result = requests.get('https://nodenormalization-sri.renci.org/get_normalized_nodes',
                            params={'curie': entity_list})
        get_label_result = get_label_result.json()
    except:
        # Catch if the call to the API fails (API not available)
        logging.info("Translator API down: https://nodenormalization-sri.renci.org/apidocs")
        get_label_result = {}
    # Response is a JSON:
    # { "HP:0007354": {
    #     "id": { "identifier": "MONDO:0004976",
    #       "label": "amyotrophic lateral sclerosis" },
    return get_label_result