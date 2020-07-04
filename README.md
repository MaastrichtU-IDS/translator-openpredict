[![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22)

**Translator OpenPredict** 🔮🐍 is API to compute and serve predicted biomedical concepts associations using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/), for the [NCATS Translator project](https://ncats.nih.gov/translator/about). This services has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.

## Install package

```bash
pip install openpredict
```

> PyPI link : https://pypi.org/project/openpredict

## Run the API

### Run in Python script 

```python
from openpredict import openpredict_api

port = 8808
debug = False
openpredict_api.start_api(port, debug)
```

> Access the Swagger UI at http://localhost:8808/ui

> Run by default in production, set `debug = True` to run in development mode. 

### Run with the command line

Run in production with [Tornado Web Server 🌪️](https://www.tornadoweb.org/en/stable/)

```bash
openpredict start-api
```

> Access the Swagger UI at http://localhost:8808/ui

Provide the port as arguments:

```bash
openpredict start-api --port 8808
```

Show help:

```bash
openpredict --help
```

### Run with docker-compose

Clone the repository:

```bash
git clone https://github.com/MaastrichtU-IDS/openpredict.git
cd openpredict
```

Start the `openpredict-api` container:

```bash
docker-compose up
```

Stop the container:

```bash
docker-compose down
```

## Compute the model

Run the pipeline to compute the model used by the API.

### From a Python script

```python
from openpredict.compute_similarities import get_drug_disease_similarities

get_drug_disease_similarities()
```

### From the commandline

```bash
openpredict compute-similarities
```

## Instructions for development

Contributions and pull requests are welcome! Follow those instructions if you want to make changes to the OpenPredict source code:

### Clone

```bash
git clone https://github.com/MaastrichtU-IDS/openpredict.git
cd openpredict
```

### Install

This will install `openpredict` and update the package automatically when the files changes locally 🔃

```bash
pip install -e .
```

### Start for development

Run in development with [Flask 🧪](https://flask.palletsprojects.com/en/1.1.x/). The API will reload automatically at each change 🔃

```bash
openpredict start-api --debug
```

### Test

Run the OpenPredict API tests locally:

```bash
pytest tests
```

### Generate documentation

See automatically generated documentation in [docs/ 📖](docs/) 

Documentation generated Using [pydoc-markdown](https://pydoc-markdown.readthedocs.io/en/latest/)

```bash
pip install pydoc-markdown
```

Generate markdown documentation page for the OpenPredict package in `docs/`

```bash
pydoc-markdown --render-toc -p openpredict > docs/README.md
```

> This can also be done using Sphinx, see this article on [deploying Sphinx to GitHub Pages](https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/)
>
> ```bash
> pip install sphinx
> sphinx-quickstart docs/ --project 'openpredict' --author 'Vincent Emonet'
> cd docs/
> make html
> ```