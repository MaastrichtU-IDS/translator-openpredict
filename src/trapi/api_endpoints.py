import os
from enum import Enum

from fastapi import APIRouter, File, UploadFile

from openpredict.predict_output import PredictOptions
from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict_model.train import add_embedding
from trapi.loaded_models import models_graph, models_list


class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


app = APIRouter()


# Generate endpoints for the loaded models
def endpoint_factory(predict_func):

    def prediction_endpoint(
        input_id: str = predict_func._trapi_predict['default_input'],
        model_id: str = predict_func._trapi_predict['default_model'],
        min_score: float = None, max_score: float = None,
        n_results: int = None
    ):
        try:
            return predict_func(input_id, PredictOptions.parse_obj({
                "model_id": model_id,
                "min_score": min_score,
                "max_score": max_score,
                "n_results": n_results,
                # "types": ['biolink:Drug'],
            }))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return (f'Error when running the prediction: {e}', 500)

    return prediction_endpoint


for loaded_model in models_list:
    for predict_func in loaded_model['endpoints']:
        app.add_api_route(
            path=predict_func._trapi_predict['path'],
            methods=["GET"],
            # endpoint=copy_func(prediction_endpoint, model['path'].replace('/', '')),
            endpoint=endpoint_factory(predict_func),
            name=predict_func._trapi_predict['name'],
            openapi_extra={"description": predict_func._trapi_predict['description']},
            response_model=dict,
            tags=["models"],
        )



@app.get("/features", name="Return the features trained in the models",
    description="""Return the features trained in the model, for Drugs, Diseases or Both.""",
    response_model=dict,
    tags=["openpredict"],
)
def get_features(embedding_type: EmbeddingTypes ='Drugs') -> dict:
    """Get features in the model

    :return: JSON with features
    """
    if type(embedding_type) is EmbeddingTypes:
        embedding_type = embedding_type.value
    return retrieve_features(models_graph, embedding_type)



@app.get("/models", name="Return the models with their training features and scores",
    description="""Return the models with their training features and scores""",
    response_model=dict,
    tags=["openpredict"],
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models(models_graph)



@app.post("/embedding", name="Upload your embedding for drugs or diseases",
    description="""Upload your embedding file:

1. Select which types do you have in the embeddings: Drugs, Diseases or Both.

2. Define the base `model_id`: use the `/models` call to see the list of trained models with their characteristics, and pick the ID of the model you will use as base to add your embedding

3. The model will be retrained and evaluation will be stored in a triplestore (available in `/models`)
""",
    response_model=dict,
    tags=["openpredict"],
)
def post_embedding(
        emb_name: str, description: str,
        types: EmbeddingTypes ='Both', model_id: str ='openpredict_baseline',
        apikey: str=None,
        uploaded_file: UploadFile = File(...)
    ) -> dict:
    """Post JSON embeddings via the API, with simple APIKEY authentication
    provided in environment variables
    """
    if type(types) is EmbeddingTypes:
        types = types.value

    # Ignore the API key check if no env variable defined (for development)
    if os.getenv('OPENPREDICT_APIKEY') == apikey or os.getenv('OPENPREDICT_APIKEY') is None:
        embedding_file = uploaded_file.file
        run_id, scores = add_embedding(
            embedding_file, emb_name, types, model_id)
        print('Embeddings uploaded')
        # train_model(False)
        return {
            'status': 200,
            'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(scores)
        }
    else:
        return {'Forbidden': 403}
