import pytest
from openpredict.compute_similarities import get_drug_disease_similarities

def test_get_drug_disease_similarities():
    """Test model to get drug-disease similarities"""
    similiraties_json = get_drug_disease_similarities()

    assert 0.80 < similiraties_json['precision'] < 0.95
    assert 0.60 < similiraties_json['recall'] < 0.80
    assert 0.80 < similiraties_json['accuracy'] < 0.95
    assert 0.85 < similiraties_json['roc_auc'] < 0.95
    assert 0.70 < similiraties_json['f1'] < 0.85
    assert 0.80 < similiraties_json['average_precision'] < 0.95