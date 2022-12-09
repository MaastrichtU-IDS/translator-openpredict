[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

Testing plan for the OpenPredict API published at https://openpredict.semanticscience.org

## Manual tests

Use the [`docs/openpredict-examples.ipynb`](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb) Jupyter notebook to manually try queries against the [OpenPredict API](https://openpredict.semanticscience.org).

## Automated testing plan

Testing of the Translator OpenPredict API is separated in 3 parts:

- **Integration**: the API is tested using integration tests, on a local API started for the tests, at every push to the `master` branch. This allows us to prevent deploying the OpenPredict API if the changes added broke some of its features
- **Production**: the API hosted in production is tested by a workflow everyday at 1:00 GMT+1, so that we are quickly notified if the production API is having an issue
- **Deployment**: a workflow tests and publish the OpenPredict API Docker image build process to insure the API can be redeployed easily

When one of those 3 workflows fails we take action to fix the source of the problem.

Requirements to run the tests: Docker

To run the test locally, you will need to first start the OpenPredict API with docker:

```bash
docker-compose up
```

### Production tests

[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml)

Integration tests are run automatically by the [GitHub Action workflow `.github/workflows/run-tests-prod.yml`](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) everyday at 01:00am GMT+1 on the [OpenPredict production API](https://openpredict.semanticscience.org)

We test for an expected number of results and a few specific results.

* POST `/query` TRAPI operation by requesting:
  * Predicted drugs for a given disease
  * Predicted diseases for a given drug
* GET `/predict` BioThings API operation by requesting:
  * Predicted drugs for a given disease
  * Predicted diseases for a given drug

Run the tests of the OpenPredict production API locally:

```bash
docker-compose exec api pytest tests/production
```

### Integration tests

[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml)

Integration tests on a local API are run automatically by the [GitHub Action workflow `.github/workflows/run-tests.yml`](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) at each push to the `master` branch.

We test the embeddings computation with a Spark local context ([setup with a GitHub Action](https://github.com/marketplace/actions/setup-apache-spark)), and without Spark context (using NumPy and pandas)

You can run all the local integration tests with docker-compose:

```bash
docker-compose run tests
```

To run a specific test in a specific file, and display `print()` lines in the output:

```bash
docker-compose run tests --entrypoint pytest tests/integration/test_openpredict_api.py::test_post_trapi -s
```

## Docker tests

[![Publish Docker image](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/publish-docker.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/publish-docker.yml)

At each new release we run the GitHub Action workflow `.github/workflows/publish-docker.yml` to test the deployment of the OpenPredict API in a Docker container, and we publish a new image for each new version of the OpenPredict API.

## Additional tests

[![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml)

We run an additional workflow which to check for vulnerabilities using the [CodeQL analysis engine](https://securitylab.github.com/tools/codeql).
