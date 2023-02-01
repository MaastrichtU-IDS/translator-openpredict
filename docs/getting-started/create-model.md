
## 🍪 Start a new prediction project

A template to help user quickly start a new prediction project with the recommended structure ([MaastrichtU-IDS/cookiecutter-openpredict-api](https://github.com/MaastrichtU-IDS/cookiecutter-openpredict-api/)). It will ask you a few questions (e.g. the name of your project), and bootstrap a repository with everything ready to start developing your prediction models.

Run these commands to install `cookiecutter` and generate your project:

```bash
pip install cookiecutter
cookiecutter https://github.com/MaastrichtU-IDS/cookiecutter-openpredict-api
```

Once your project has been generated, follow the instructions in the generated `README.md` to run your project in development.

## 💾 Save a generated model

A helper function is provided to easily save a generated model, its metadata, and the data used to generate it. It uses tools such as [`dvc`](https://dvc.org/) and [`mlem`](https://mlem.ai/) to store large model outside of the git repository.

```python
from openpredict import save

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

## 🔮 Define the prediction endpoint

The `openpredict` package provides a decorator `@trapi_predict` to annotate your functions that generate predictions. The code for this package is in `src/openpredict/`.

Predictions generated from functions decorated with `@trapi_predict` can easily be imported in the Translator OpenPredict API, exposed as an API endpoint to get predictions from the web, and queried through the Translator Reasoner API (TRAPI)

```python
from openpredict import trapi_predict, PredictOptions, PredictOutput

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
