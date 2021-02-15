[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Test%20production%20API/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Test+production+API%22) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/CodeQL%20analysis/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22CodeQL+analysis%22)

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

Testing plan for the OpenPredict API published at https://openpredict.semanticscience.org

## Manual tests

Use the [`docs/openpredict-examples.ipynb`](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb) Jupyter notebook to manually try queries against the OpenPredict API.

## Automated tests

Install the required dependency to run tests:

```bash
pip install pytest
```

### Integration tests

Integration tests are run automatically by a [GitHub Action workflow](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) everyday at 01:00am GMT+1 on the OpenPredict production API.

We test for an expected number of results and a few specific results.

* POST `/query` TRAPI operation by requesting:
  * Predicted drugs for a given disease
  * Predicted diseases for a given drug
* GET `/predict` BioThings API operation by requesting:
  * Predicted drugs for a given disease
  * Predicted diseases for a given drug

To run the tests of the OpenPredict production API locally:

```bash
pytest tests/integration
```

### Unit tests

Unit tests on a local API are run automatically by a [GitHub Action workflow](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) at each push to the `master` branch ✔️

Run tests for the different components of OpenPredict locally:

```bash
pytest tests/unit
```

Run a specific test in a specific file, and display `print()` lines in the output:

```bash
pytest tests/unit/test_openpredict_api.py::test_post_trapi -s
```

## Docker tests

At each new release we run a GitHub Action workflow to test the deployment of the OpenPredict API in a Docker container, and we publish a new image for each new version of the OpenPredict API.

## Known issues

Facing issue with `pytest` install even using virtual environments? Try this solution:

```bash
python3 -m pip install -e .
python3 -m pip install pytest
python3 -m pytest
```
