import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlem import api as mlem
from rdflib import Graph

# from mlem.api import save as mlem_save, load as mlem_load
from openpredict.rdf_utils import get_run_metadata
from openpredict.utils import log


@dataclass
class LoadedModel():
    model: Any
    metadata: Graph
    features: Any = None


def save(
    model: Any,
    path: str,
    sample_data: Any,
    scores: Optional[Dict] = None,
    features: Optional[Any] = None,
    hyper_params: Optional[Dict] = None,
) -> None:
    model_name = path.rsplit('/', 1)[-1]
    # print(os.path.isabs(path))
    # if not os.path.isabs(path):
    #     path = os.path.join(os.getcwd(), path)
    log.info(f"💾 Saving model in {path}")

    mlem.save(model, path, sample_data=sample_data)
    # pickle.dump(model, open(f"{path}", "wb"))

    if features:
        with open(f"{path}.features", "wb") as f:
            pickle.dump(features, f)

    g = get_run_metadata(scores, features, hyper_params, model_name)
    g.serialize(f"{path}.ttl", format='ttl')
    g.serialize(f"{path}.json", format='json-ld')

    # TODO: generate and store RDF metadata

    # run_id = add_run_metadata(scores, model_features,
    #                           hyper_params, run_id=run_id)
    # added_feature_uri = add_feature_metadata(emb_name, description, types)
    # types = Drug/Disease


def load(path: str) -> LoadedModel:
    log.info(f"Loading model from {path}")
    model = mlem.load(path)
    # model = pickle.load(open(path, "rb"))

    features = None
    features_path = f"{path}.features"
    if os.path.exists(features_path):
        features = pickle.load(open(features_path, "rb"))

    g = Graph()
    g.parse(f"{path}.json", format='json-ld')

    return LoadedModel(model=model, features=features, metadata=g)
