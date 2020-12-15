[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) [![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/CodeQL%20analysis/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22CodeQL+analysis%22)

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

# Contributing

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/MaastrichtU-IDS/translator-openpredict/issues) if applicable.

If you are part of the [MaastrichtU-IDS organization on GitHub](https://github.com/MaastrichtU-IDS) you can directly create a branch in this repository. Otherwise you will need to first [fork this repository](https://github.com/MaastrichtU-IDS/translator-openpredict/fork).

To contribute:

1. Clone the repository 📥

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

2. Create a new branch from the `master` branch and add your changes to this branch 🕊️

```bash
git checkout -b my-branch
```

## Development process

Install `openpredict` from the source code, and update the package automatically when the files changes locally :arrows_counterclockwise:

```bash
pip3 install -e .
```

> See the [main README](https://github.com/MaastrichtU-IDS/translator-openpredict) for more details on the package installation.

The OpenPredict API store its metadata using RDF:

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

### Start OpenPredict API with a local Virtuoso triplestore 🗄️

To store OpenPredict metadata in a local triplestore:

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


### Reset your local OpenPredict data

If you want to reset the (meta)data used by OpenPredict locally:

1. Stop OpenPredict API
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

## Run tests ✔️

[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22)

Tests are automatically run by a [GitHub Action](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) at each push to the `master` branch. They are also run in the GitHub Action to publish a package.

Run the **OpenPredict API** tests locally:

```bash
pytest tests
```

Run a specific test in a specific file, and display `print()` lines in the output:

```bash
pytest tests/test_openpredict_api.py::test_post_reasoner_predict -s
```

## Generate pydoc for the code 📖

Documentation in [docs/README-pydoc.md](https://github.com/MaastrichtU-IDS/translator-openpredict/tree/master/docs/README-pydoc.md) generated from the Python source code docstrings using [pydoc-markdown](https://pydoc-markdown.readthedocs.io/en/latest/).

```bash
pip3 install pydoc-markdown
```

Generate markdown documentation page for the `openpredict` package in `docs/`

```bash
pydoc-markdown --render-toc -p openpredict > docs/README-pydoc.md
```

Modify the generated page title:

```bash
find docs/README-pydoc.md -type f -exec sed -i "s/# Table of Contents/# OpenPredict Package documentation 🔮🐍/g" {} +
```

> **❌ Currently not used**: this can also be done using Sphinx, see this article on [deploying Sphinx to GitHub Pages](https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/)
>
> ```bash
> pip3 install sphinx
> sphinx-quickstart sphinx-docs/ --project 'openpredict' --author 'Vincent Emonet'
> cd sphinx-docs/
> make html
> ```

## Pull Request process

1. Ensure the tests are passing before sending a pull request 🧪

2. Update the `README.md` with details of changes, this includes new environment variables, exposed ports, useful file locations and container parameters 📝
3. [Send a pull request](https://github.com/MaastrichtU-IDS/translator-openpredict/compare) to the `master` branch, answer the questions in the pull request message 📤
4. Project contributors will review your change as soon as they can ✔️

## Versioning process

The versioning scheme for new releases on GitHub used is [SemVer](http://semver.org/) (Semantic Versioning).

Change version in `setup.py` before new release.