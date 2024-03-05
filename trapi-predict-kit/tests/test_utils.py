from trapi_predict_kit import get_entity_types, get_run_metadata, normalize_id_to_translator, resolve_entities

from .conftest import get_predictions


def test_resolve_entities():
    """Test the function to resolve entities using the Name Resolution API"""
    resp = resolve_entities("alzheimer")
    curie_list = [item["curie"] for item in resp]
    assert "MONDO:0004975" in curie_list


def test_normalize_id_to_translator():
    to_convert = "OMIM:104300"
    normalized = normalize_id_to_translator([to_convert])
    assert normalized[to_convert] == "MONDO:0007088"


def test_get_entity_types():
    types = get_entity_types("OMIM:104300")
    assert "biolink:Disease" in types


def test_get_run_metadata():
    g = get_run_metadata({}, [], {})
    assert len(g) > 0


def test_trapi_predict_decorator():
    res = get_predictions({"subjects": ["drugbank:DB00002"]})
    assert get_predictions._trapi_predict["edges"][0]["subject"] == "biolink:Drug"
    assert len(res["hits"]) == 1
