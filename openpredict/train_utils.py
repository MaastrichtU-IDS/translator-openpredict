import logging
import pkg_resources
from datetime import datetime
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, XSD, DC, DCTERMS, VOID

TTL_METADATA_FILE = pkg_resources.resource_filename('openpredict', 'data/openpredict-metadata.ttl')
OPENPREDICT_NAMESPACE = 'https://w3id.org/openpredict/'
MLS_NAMESPACE = 'http://www.w3.org/ns/mls#'

def generate_classifier_metadata(classifier_id, scores, label="OpenPredict classifier"):
    """Generate RDF metadata for a classifier and save it in data/openpredict-metadata.ttl, based on OpenPredict model:
    https://github.com/fair-workflows/openpredict/blob/master/data/rdf/results_disjoint_lr.nq

    :param classifier_id: Unique ID for the classifier
    :param scores: scores
    :param label: label of the classifier
    :return: predictions in array of JSON object
    """
    g = Graph()
    g.parse(TTL_METADATA_FILE, format="ttl")

    clf_uri = URIRef(OPENPREDICT_NAMESPACE + classifier_id)
    clf_prop_prefix = OPENPREDICT_NAMESPACE + classifier_id + "/"

    if not (clf_uri, None, None) in g:
        print('Generating RDF metadata for the trained classifier at ' + TTL_METADATA_FILE)
        g.add((clf_uri, RDF.type, URIRef(MLS_NAMESPACE + 'ModelEvaluation')))
        g.add((clf_uri, RDFS.label, Literal(label)))
        g.add((clf_uri, URIRef('http://www.w3.org/ns/prov#generatedAtTime'), Literal(datetime.now(), datatype=XSD.dateTime)))

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