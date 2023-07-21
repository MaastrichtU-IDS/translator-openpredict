## 💾 Save a generated model

Once you have setup your project it is time to start defining your model training. We recommend to do this in a specific file, e.g. `train.py`

A helper function is provided to easily save a generated model, its metadata, and the data used to generate it. It uses tools such as [`dvc`](https://dvc.org/) and [`mlem`](https://mlem.ai/) to store large model outside of the git repository. Here is an example:

```python
from trapi_predict_kit import save

hyper_params = {
    'penalty': 'l2',
    'dual': False,
    'tol': 0.0001,
    'C': 1.0,
    'random_state': 100
}

saved_model = save(
    model=clf,
    path="models/my_model",
    sample_data=sample_data,
    hyper_params=hyper_params,
    scores=scores,
)
```

If you generated a project from the template you will find it in the `train.py` script.

⚠️ Once you have trained your model don't forget to add it, usually in the `models/` folder, and push it with `dvc` (along with all the data required to train the model in the `data/` folder)

## 🔮 Define the prediction endpoint

Once your model has been trained you can create a function taking an input ID and generating predictions for it. We recommend to do it in a specific file, e.g. `predict.py`

The `openpredict` package provides a decorator `@trapi_predict` to annotate your functions that generate predictions. Predictions generated from functions decorated with `@trapi_predict` can easily be imported in the Translator OpenPredict API, exposed as an API endpoint to get predictions from the web, and queried through the Translator Reasoner API (TRAPI).

The annotated predict functions are expected to take 2 input arguments: the input ID (string) and options for the prediction (dictionary). And it should return a dictionary with a list of predicted associated entities hits (see below for a practical example)

 Here is an example:

```python
from trapi_predict_kit import trapi_predict, PredictOptions, PredictOutput

@trapi_predict(path='/predict',
    name="Get predicted targets for a given entity",
    description="Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.",
    edges=[
        {
            'subject': 'biolink:Drug',
            'predicate': 'biolink:treats',
            'object': 'biolink:Disease',
        },
        {
            'subject': 'biolink:Disease',
            'predicate': 'biolink:treated_by',
            'object': 'biolink:Drug',
        },
    ],
	nodes={
        "biolink:Disease": {
            "id_prefixes": [
                "OMIM"
            ]
        },
        "biolink:Drug": {
            "id_prefixes": [
                "DRUGBANK"
            ]
        }
    }
)
def get_predictions(
        input_id: str, options: PredictOptions
    ) -> PredictOutput:
    # Add the code the load the model and get predictions here
    predictions = {
        "hits": [
            {
                "id": "DB00001",
                "type": "biolink:Drug",
                "score": 0.12345,
                "label": "Leipirudin",
            }
        ],
        "count": 1,
    }
    return predictions
```

If you generated a project from the template you will find it in the `predict.py` script.
