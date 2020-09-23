import logging
import pkg_resources
import uuid 
from datetime import datetime
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, XSD, DC, DCTERMS, VOID

TTL_METADATA_FILE = pkg_resources.resource_filename('openpredict', 'data/openpredict-metadata.ttl')
OPENPREDICT_NAMESPACE = 'https://w3id.org/openpredict/'
MLS_NAMESPACE = 'http://www.w3.org/ns/mls#'
BIOLINK_NAMESPACE = 'https://w3.org/biolink/'

def generate_feature_metadata(id, description, type):
    """Generate RDF metadata for a feature

    :param id: if used to identify the feature
    :param description: feature description
    :param type: feature type
    :return: rdflib graph after loading the feature
    """
    g = Graph()
    g.parse(TTL_METADATA_FILE, format="ttl")

    feature_uri = URIRef(OPENPREDICT_NAMESPACE + 'feature/' + id)
    g.add((feature_uri, RDF.type, URIRef(MLS_NAMESPACE + 'Feature')))
    g.add((feature_uri, DC.identifier, Literal(id)))
    g.add((feature_uri, DC.description, Literal(description)))
    g.add((feature_uri, URIRef(OPENPREDICT_NAMESPACE + 'embedding_type'), Literal(type)))
    g.add((feature_uri, URIRef('http://www.w3.org/ns/prov#generatedAtTime'), Literal(datetime.now(), datatype=XSD.dateTime)))

    g.serialize(TTL_METADATA_FILE, format="ttl")
    return g

def generate_classifier_metadata(scores, model_features, label="OpenPredict classifier"):
    """Generate RDF metadata for a classifier and save it in data/openpredict-metadata.ttl, based on OpenPredict model:
    https://github.com/fair-workflows/openpredict/blob/master/data/rdf/results_disjoint_lr.nq

    :param scores: scores
    :param model_features: List of features in the model
    :param label: label of the classifier
    :return: predictions in array of JSON object
    """

    classifier_id = uuid.uuid1() 
    g = Graph()
    g.parse(TTL_METADATA_FILE, format="ttl")

    clf_uri = URIRef(OPENPREDICT_NAMESPACE + 'model/' + classifier_id)
    clf_prop_prefix = OPENPREDICT_NAMESPACE + classifier_id + "/"

    if not (clf_uri, None, None) in g:
        print('Generating RDF metadata for the trained classifier at ' + TTL_METADATA_FILE)
        g.add((clf_uri, RDF.type, URIRef(MLS_NAMESPACE + 'ModelEvaluation')))
        g.add((clf_uri, RDFS.label, Literal(label)))
        g.add((clf_uri, URIRef('http://www.w3.org/ns/prov#generatedAtTime'), Literal(datetime.now(), datatype=XSD.dateTime)))

        for feature in model_features:
            feature_uri = URIRef(OPENPREDICT_NAMESPACE + 'feature/' + feature)
            g.add((clf_uri, URIRef(OPENPREDICT_NAMESPACE + 'has_features'), feature_uri))

        for key in scores.keys():
            key_uri = URIRef(clf_prop_prefix + key)
            g.add((clf_uri, URIRef(MLS_NAMESPACE + 'specifiedBy'), key_uri))
            g.add((key_uri, RDF.type, URIRef(MLS_NAMESPACE + 'EvaluationMeasure')))
            g.add((key_uri, RDFS.label, Literal(key)))
            g.add((key_uri, URIRef(MLS_NAMESPACE + 'hasValue'), Literal(scores[key], datatype=XSD.double)))

        # g.serialize("data/openpredict-metadata.ttl", format="ttl")
        g.serialize(TTL_METADATA_FILE, format="ttl")
    else:
        print('RDF metadata already generated for this classifier')

    # import pprint
    # for stmt in g:
    #     pprint.pprint(stmt)

    return g