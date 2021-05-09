[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml) [![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml)

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

# Contributing

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/MaastrichtU-IDS/translator-openpredict/issues) if applicable.

If you are part of the [MaastrichtU-IDS organization on GitHub](https://github.com/MaastrichtU-IDS) you can directly create a branch in this repository. Otherwise you will need to first [fork this repository](https://github.com/MaastrichtU-IDS/translator-openpredict/fork).

To contribute:

1. Clone the repository (change the URL for your fork) 📥

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

2. Create a new branch from the `master` branch and add your changes to this branch 🕊️

```bash
git checkout -b my-branch
```

## Development process 👩‍💻

Install `openpredict` from the source code, and update the package automatically when the files changes locally :arrows_counterclockwise:

```bash
pip3 install -e .
```

> See the [main README](https://github.com/MaastrichtU-IDS/translator-openpredict) for more details on the package installation.

The OpenPredict API stores its metadata using RDF:

* We use a `.ttl` file in `data/` in local development
* It can use the open source [Virtuoso triplestore](https://virtuoso.openlinksw.com/) in local development environment with docker
* We use [Ontotext GraphDB](https://github.com/Ontotext-AD/graphdb-docker) in production at IDS, but you are free to use any other triplestore!

### Start the OpenPredict API :rocket:


Start the **OpenPredict API in debug mode** on http://localhost:8808 (the API will be reloaded automatically at each change to the code)

```bash
openpredict start-api --debug
```

> OpenPredict metadata will be stored in `.ttl` RDF files

Start the **OpenPredict API in productionn mode** with Tornado (the API will not reload at each change)

```bash
openpredict start-api
```

### Optional: start OpenPredict API with a local Virtuoso triplestore 🗄️

To test OpenPredict using a local triplestore to store metadata:

1. **Start the Virtuoso triplestore** locally on http://localhost:8890 using Docker (login: `dba` / `dba`):

```bash
docker-compose -f docker-compose.dev.yml up -d --force-recreate
```

2. Start the **OpenPredict API in debug mode** on http://localhost:8808 (the API will be reloaded automatically at each change to the code)

```bash
openpredict start-api --debug
```

3. **Stop** the Virtuoso container:

```bash
docker-compose down
```


### Reset your local OpenPredict data 🗑️

If you want to reset the (meta)data used by OpenPredict locally:

1. Stop OpenPredict API (and Virtuoso if used).
2. Use the `reset_openpredict.sh` script to delete the folders where the OpenPredict API and Virtuoso data are stored, in `data/virtuoso` and `data/openpredict`

```bash
./reset_openpredict.sh
```

> This command uses `sudo` to be able to delete the `data/virtuoso` folder which has been created by the `docker` user.
>
> On Windows: delete all files in `data` folder, just keep `initial-openpredict-metadata.ttl` 

See more **[documentation to deploy the OpenPredict API](https://github.com/MaastrichtU-IDS/translator-openpredict/tree/master/docs)** locally or with Docker.

## Create a new API call 📝

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

## Run tests 🧪

[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22)

Check the [TESTING.md](/TESTING.md) documentation.

## Pull Request process 📬

1. Ensure the tests are passing before sending a pull request 🧪

2. Update the `README.md` with details of changes, this includes new environment variables, exposed ports, useful file locations and container parameters 📝
3. [Send a pull request](https://github.com/MaastrichtU-IDS/translator-openpredict/compare) to the `master` branch, answer the questions in the pull request message 📤
4. Project contributors will review your change as soon as they can ✔️

---

## Additional informations about releases ℹ️

This part is not required to be completed if you are looking into contributing, it is purely informative on the release process of the OpenPredict API.

### Release process 🏷️

The versioning scheme for new releases on GitHub used is [SemVer](http://semver.org/) (Semantic Versioning).

1. Change version in `setup.py` before new a release, e.g. `0.0.7`
2. Create a new release in the [GitHub web UI](https:///github.com/MaastrichtU-IDS/translator-openpredict).Provide the version as tag, e.g. `v0.0.7`
3. When you publish the new release, a [GitHub Action workflow](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) will be automatically run to run the tests, and publish the `openpredict` package to [PyPI](https://pypi.org/project/openpredict/).

### Publish a new Docker image 📦

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

### Generate pydoc for the code 📖

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