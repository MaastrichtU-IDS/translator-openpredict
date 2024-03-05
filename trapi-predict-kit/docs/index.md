**TRAPI Predict Kit** is a python package that helps data scientists to build, and **publish prediction models** in a [FAIR](https://www.go-fair.org/fair-principles/) and reproducible manner. It provides helpers for various steps of the process:

* A template to help user quickly bootstrap a new prediction project with the recommended structure ([MaastrichtU-IDS/cookiecutter-trapi-predict-kit](https://github.com/MaastrichtU-IDS/cookiecutter-trapi-predict-kit/))
* Helper function to easily save a generated model, its metadata, and the data used to generate it. It uses tools such as [`dvc`](https://dvc.org/) and [`mlem`](https://mlem.ai/) to store large model outside of the git repository.
* Deploy API endpoints for retrieving predictions, which comply with the NCATS Biomedical Data Translator standards ([Translator Reasoner API](https://github.com/NCATSTranslator/ReasonerAPI) and [BioLink model](https://github.com/biolink/biolink-model)), using a decorator `@trapi_predict` to simply annotate the function that produces predicted associations for a given input entity

Predictions are usually generated from machine learning models (e.g. predict disease treated by drug), but it can adapt to generic python function, as long as the input params and return object follow the expected structure.

## Installation

To start a new project to develop and publish models for predictions we recommend to use the cookiecutter template that bootstrap, see the page [create a model](getting-started/create-model) for more details.

Otherwise the `trapi-predict-kit` package can be installed with `pip`:

```bash
pip install trapi-predict-kit
```

## Acknowledgments​

This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project. And Predictions of the OpenPredict API made using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/).

This service is funded by the [NIH NCATS Translator project](https://ncats.nih.gov/translator/about).

![Funded the the NIH NCATS Translator project](https://ncats.nih.gov/files/TranslatorGraphic2020_1100x420.jpg)
