import pytest
from openpredict.openpredict_omim_drugbank import build_drug_disease_classifier

def test_build_drug_disease_classifier():
    """Test the model to get drug-disease similarities (drugbank-omim)"""
    clf, scores = build_drug_disease_classifier()

    assert 0.80 < scores['precision'] < 0.95
    assert 0.60 < scores['recall'] < 0.80
    assert 0.80 < scores['accuracy'] < 0.95
    assert 0.85 < scores['roc_auc'] < 0.95
    assert 0.70 < scores['f1'] < 0.85
    assert 0.80 < scores['average_precision'] < 0.95
