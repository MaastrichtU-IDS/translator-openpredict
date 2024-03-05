To make your models and experiment more reusable we recommend to properly save the trained model.

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
