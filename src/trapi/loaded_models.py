from openpredict.rdf_utils import get_loaded_graph
from openpredict_model.predict import get_predictions, get_similarities

# models_list = [get_predictions, get_similarities]

models_list = [
    {
        "model": "models/openpredict_baseline",
        "endpoints": [get_predictions, get_similarities]
    },
]

models_graph = get_loaded_graph(models_list)

# models_list = [get_predictions]
