from dataclasses import dataclass
from typing import Any, Optional

# from fairworkflows import is_fairstep
from mlem import api as mlem
from rdflib import Graph

# from mlem.api import save as mlem_save, load as mlem_load
from openpredict.rdf_utils import get_run_metadata
from openpredict.utils import log


@dataclass
class LoadedModel():
    path: str
    model: Any
    metadata: Graph
    hyper_params: Optional[Any] = None
    scores: Optional[Any] = None
    # features: Any = None


# @is_fairstep(label='Save a model')
def save(
    model: Any,
    path: str,
    sample_data: Any,
    scores: Optional[Any] = None,
    hyper_params: Optional[Any] = None,
    # model: Any,
    # path: str,
    # sample_data: Any,
    # scores: Optional[Dict] = None,
    # hyper_params: Optional[Dict] = None,
) -> LoadedModel:
    model_name = path.rsplit('/', 1)[-1]
    # print(os.path.isabs(path))
    # if not os.path.isabs(path):
    #     path = os.path.join(os.getcwd(), path)
    log.info(f"💾 Saving model in {path}")

    # mlem_model = MlemModel.from_obj(model, sample_data=sample_data)
    # mlem_model.dump(path)
    # print(mlem_model)
    mlem.save(model, path, sample_data=sample_data)

    g = get_run_metadata(scores, sample_data, hyper_params, model_name)
    g.serialize(f"{path}.ttl", format='ttl')
    # os.chmod(f"{path}.ttl", 0o644)
    # os.chmod(f"{path}.mlem", 0o644)

    # TODO: generate and store RDF metadata
    # return path
    return LoadedModel(
        path=path,
        model=model,
        metadata=g,
        hyper_params=hyper_params,
        scores=scores,
    )



def load(path: str) -> LoadedModel:
    log.info(f"Loading model from {path}")
    model = mlem.load(path)

    g = Graph()
    g.parse(f"{path}.ttl", format='ttl')

    # TODO: extract scores and hyper_params from RDF

    return LoadedModel(path=path, model=model, metadata=g)
