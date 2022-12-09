[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml) [![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml)

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

# Contributing

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/MaastrichtU-IDS/translator-openpredict/issues) if applicable.

If you are part of the [MaastrichtU-IDS organization on GitHub](https://github.com/MaastrichtU-IDS) you can directly create a branch in this repository. Otherwise you will need to first [fork this repository](https://github.com/MaastrichtU-IDS/translator-openpredict/fork).

## 👩‍💻 Development process

To work with translator-openpredict locally:

### 📥️ Install

1. Clone the repository:

   ```bash
   git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
   cd translator-openpredict
   ```

2. Pull the data required to run the models in the `data` folder with [`dvc`](https://dvc.org/):

   ```bash
   pip install dvc
   dvc pull
   ```

### 🚀 Start the API

Start the API in development with docker, the API will automatically reload when you make changes in the code:

```bash
docker-compose up api
```

You will need to re-build the docker image if you add new dependencies to the `pyproject.toml`:

```bash
docker-compose up api --build
```

### 🧪 Run tests

[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22)

You can run the tests locally after starting the API wit docker-compose:

```bash
docker-compose exec api pytest tests/integration
```

See the [`TESTING.md`](/TESTING.md) file for more details on testing the API.

### 📤️ Push changes to the data

If you make changes to the data in the `data` folder you will need to add and push this data on [DagsHub](https://dagshub.com/docs/integration_guide/dvc/) with `dvc` 

1. Go to [dagshub.com](https://dagshub.com/user/login), and login with GitHub or Google

2. Get your token and set your credentials:

   ```bash
   export DAGSHUB_USER="vemonet"
   export DAGSHUB_TOKEN="TOKEN"
   ```

3. Connect your local repository with the created DagsHub project:

   ```bash
   dvc remote add origin https://dagshub.com/vemonet/translator-openpredict.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user $DAGSHUB_USER
   dvc remote modify origin --local password $DAGSHUB_TOKEN
   ```

4. Push data:

   ```bash
   dvc push
   ```

> ⚠️ Open source projects on DagsHub using the free plan have a 10G storage limit.

## 📝 Create a new API call

Guidelines to create a new API  call in the OpenPredict Open API.

1. Create the operations in the [openpredict/openapi.yml](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/openpredict/openapi.yml#L44) file

Provide the path to the function that will resolve this API call in `operationId`:

```yaml
paths:
  /predict:
    get:
      operationId: openpredict.openpredict_api.get_predict
      parameters:
      - name: entity
        in: query
        description: CURIE of the entity to process (e.g. drug, disease, etc)
        example: DRUGBANK:DB00394
        required: true
        schema:
          type: string
```

2. Now, create the function in the [openpredict/openpredict_api.py](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/openpredict/openpredict_api.py#L67) file

```python
def get_predict(entity='DB00001'):
    print("Do stuff with " + entity)
```

> The parameters provided in `openapi.yml` and the arguments of the function in `openpredict_api.py` need to match!

## 📬 Pull Request process

1. Ensure the tests are passing before sending a pull request 🧪

2. Update the `README.md` with details of changes, this includes new environment variables, exposed ports, useful file locations and container parameters 📝
3. [Send a pull request](https://github.com/MaastrichtU-IDS/translator-openpredict/compare) to the `master` branch, answer the questions in the pull request message 📤
4. Project contributors will review your change as soon as they can ✔️

---

## ℹ️ Additional informations about releases

This part is not required to be completed if you are looking into contributing, it is purely informative on the release process of the OpenPredict API.

### 🏷️ Release process

The versioning scheme for new releases on GitHub used is [SemVer](http://semver.org/) (Semantic Versioning).

1. Change version in `setup.py` before new a release, e.g. `0.0.7`
2. Create a new release in the [GitHub web UI](https:///github.com/MaastrichtU-IDS/translator-openpredict).Provide the version as tag, e.g. `v0.0.7`
3. When you publish the new release, a [GitHub Action workflow](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) will be automatically run to run the tests, and publish the `openpredict` package to [PyPI](https://pypi.org/project/openpredict/).

### 📦 Publish a new Docker image

When publishing a new version of OpenPredict we usually also publish an updated Docker image to the [MaastrichtU-IDS GitHub Container Registry](https://github.com/orgs/MaastrichtU-IDS/packages/container/package/openpredict-api).

> Replace the `latest` tag by your version number, e.g. `v0.0.7`

Build the OpenPredict API Docker image:

```bash
docker build -t ghcr.io/maastrichtu-ids/openpredict-api:latest .
```

Push to the [MaastrichtU-IDS GitHub Container Registry](https://github.com/orgs/MaastrichtU-IDS/packages/container/package/openpredict-api)

```bash
docker push ghcr.io/maastrichtu-ids/openpredict-api:latest
```

### 📖 Generate pydoc for the code

Documentation in [docs/README-pydoc.md](https://github.com/MaastrichtU-IDS/translator-openpredict/tree/master/docs/README-pydoc.md) is generated from the Python source code doc strings using [pydoc-markdown](https://pydoc-markdown.readthedocs.io/en/latest/).

```bash
pip3 install pydoc-markdown
```

Generate markdown documentation page for the `openpredict` package in `docs/`

```bash
pydoc-markdown --render-toc -p openpredict > docs/README-pydoc.md
```

Modify the generated page title automatically:

```bash
find docs/README-pydoc.md -type f -exec sed -i "s/# Table of Contents/# OpenPredict Package documentation 🔮🐍/g" {} +
```

### Update the TRAPI version

Get the latest TRAPI YAML: https://github.com/NCATSTranslator/ReasonerAPI/blob/master/TranslatorReasonerAPI.yaml

1. Update description of the service
2. Add additional calls exclusive to OpenPredict
3. Add `operationId` for each call
4. In `components:` add `schemas: QueryOptions`