import os
from enum import Enum

from fastapi import APIRouter, File, UploadFile

from openpredict.predict_output import PredictOptions
from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict_model.train import addEmbedding
from trapi.loaded_models import models_list


class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


app = APIRouter()


# Generate endpoints for the loaded models
def endpoint_factory(prediction_func, model):

    def prediction_endpoint(
        input_id: str = model['default_input'],
        model_id: str = model['default_model'],
        min_score: float = None, max_score: float = None,
        n_results: int = None
    ):
        try:
            return prediction_func(input_id, PredictOptions.parse_obj({
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


for (do_prediction, model) in models_list:
    app.add_api_route(
        path=model['path'],
        methods=["GET"],
        # endpoint=copy_func(prediction_endpoint, model['path'].replace('/', '')),
        endpoint=endpoint_factory(do_prediction, model),
        name=model['name'],
        openapi_extra={"description": model['description']},
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
    return retrieve_features(embedding_type)



@app.get("/models", name="Return the models with their training features and scores",
    description="""Return the models with their training features and scores""",
    response_model=dict,
    tags=["openpredict"],
)
def get_models() -> dict:
    """Get models with their scores and features

    :return: JSON with models and features
    """
    return retrieve_models()



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
        types: EmbeddingTypes ='Both', model_id: str ='openpredict-baseline-omim-drugbank',
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
        run_id, scores = addEmbedding(
            embedding_file, emb_name, types, description, model_id)
        print('Embeddings uploaded')
        # train_model(False)
        return {
            'status': 200,
            'message': 'Embeddings added for run ' + run_id + ', trained model has scores ' + str(scores)
        }
    else:
        return {'Forbidden': 403}
