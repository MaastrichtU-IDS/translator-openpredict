**OpenPredict** is a python package that helps data scientists to build, and **publish prediction models** in a [FAIR](https://www.go-fair.org/fair-principles/) and reproducible manner. It provides helpers for various steps of the process:

* A template to help user quickly bootstrap a new prediction project with the recommended structure ([MaastrichtU-IDS/cookiecutter-openpredict-api](https://github.com/MaastrichtU-IDS/cookiecutter-openpredict-api/))
* Helper function to easily save a generated model, its metadata, and the data used to generate it. It uses tools such as [`dvc`](https://dvc.org/) and [`mlem`](https://mlem.ai/) to store large model outside of the git repository.
* Deploy API endpoints for retrieving predictions, which comply with the NCATS Biomedical Data Translator standards ([Translator Reasoner API](https://github.com/NCATSTranslator/ReasonerAPI) and [BioLink model](https://github.com/biolink/biolink-model)), using a decorator `@trapi_predict` to simply annotate the function that produces predicted associations for a given input entity

Predictions are usually generated from machine learning models (e.g. predict disease treated by drug), but it can adapt to generic python function, as long as the input params and return object follow the expected structure.

The package can be installed with `pip`:

```bash
pip install openpredict
```
