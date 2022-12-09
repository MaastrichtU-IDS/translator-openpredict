import os
from enum import Enum

from fastapi import APIRouter, File, UploadFile

from openpredict.rdf_utils import retrieve_features, retrieve_models
from openpredict_model.train import addEmbedding


class EmbeddingTypes(str, Enum):
    Both = "Both"
    Drugs = "Drugs"
    Diseases = "Diseases"


app = APIRouter()


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
        print(emb_name, types)
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