[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) 

**Translator OpenPredict** 🔮🐍 is an API to compute and serve predicted biomedical concepts associations, for the [NCATS Translator project](https://ncats.nih.gov/translator/about). 

This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.

# Use the API 🌐

The Translator OpenPredict API serves predictions of biomedical concepts associations (e.g. disease treated by drug). 

> Feel free to try it at **[openpredict.137.120.31.102.nip.io](https://openpredict.137.120.31.102.nip.io)**

### How to use it

The user provides a drug 💊 or a disease 🦠 identifier as a CURIE (e.g. DRUGBANK:DB00394, OMIM:246300), and choose a prediction model (only OMIM - DrugBank at the moment).

* The `/predict` call will return the list of predicted target for this entity, the labels are resolved using the [Translator Name Resolver API](http://robokop.renci.org:2434/docs#/lookup/lookup_curies_lookup_post):

```json
{
  "count": 300,
  "relation": "biolink:treated_by",
  "results": [
    {
      "score": 0.8361061489249737,
      "source": {
        "id": "DRUGBANK:DB00394",
        "label": "beclomethasone dipropionate",
        "type": "drug"
      },
      "target": {
        "id": "OMIM:246300",
        "label": "leprosy, susceptibility to, 3",
        "type": "disease"
      }
    }
  ]
}
```

> Try it at https://openpredict.137.120.31.102.nip.io/predict?entity=DRUGBANK:DB00394

* The `/query` call will return the same predictions in the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) format, based on a [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) query
* The `/predicates` call will return the entities and relations returned by this API (following the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) specifications)

### Notebook example

You can find a Jupyter Notebook with [examples to query the API on GitHub](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)

# Deploy your API 📦

You can also use our tools to build new classifiers, and deploy your OpenPredict API

### Install OpenPredict 📥

You might want to use a virtual environment for Python 3.7 to isolate the installation:

```bash
# Create the virtual environment in your workspace
python3 -m venv .venv
# Activate it
source .venv/bin/activate
```

Install the latest release published on [PyPI 🏷️](https://pypi.org/project/openpredict) (or see below to [run the API with Docker](#option-3-run-with-docker))

```bash
pip3 install openpredict
```

> Package available on PyPI: [https://pypi.org/project/openpredict](https://pypi.org/project/openpredict)

---

### Train the model 🚅

Run the pipeline to train the model used by the OpenPredict API.

From a Python script:

```python
from openpredict.openpredict_omim_drugbank import train_drug_disease_classifier

train_drug_disease_classifier()
```

Or using the command line:

```bash
openpredict train-model
```

> Work in progress
>
> model_factory = 

---

### Run the API ⚙️

The API can be run different ways:

#### Option 1: Run from the command line

Use the `openpredict` CLI to start the API using the built classifiers:

```bash
openpredict start-api
```

> Access the Swagger UI at [http://localhost:8808](http://localhost:8808)

Provide the port as arguments:

```bash
openpredict start-api --port 8808
```

#### Option 2: Run from a Python script

```python
from openpredict import openpredict_api

debug = False
openpredict_api.start_api(8808, debug)
```

> Access the Swagger UI at [http://localhost:8808](http://localhost:8808)

> Run by default in production, set `debug = True` to run in development environments. 

#### Option 3: Run with Docker

Running using Docker can be convenient if you just want to run the API without installing the package locally, or if it runs in production alongside other services.

Clone the [repository](https://github.com/MaastrichtU-IDS/translator-openpredict):

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

Build and start the `openpredict-api` container with [docker-compose 🐳](https://docs.docker.com/compose/)

```bash
docker-compose up -d
```

> Access the Swagger UI at [http://localhost:8808](http://localhost:8808)

> We use [nginx-proxy](https://github.com/nginx-proxy/nginx-proxy) and [docker-letsencrypt-nginx-proxy-companion](https://github.com/nginx-proxy/docker-letsencrypt-nginx-proxy-companion) as reverse proxy for HTTP and HTTPS in production. You can change the proxy URL and port via environment variables `VIRTUAL_HOST`, `VIRTUAL_PORT` and `LETSENCRYPT_HOST` in the [docker-compose.yml](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docker-compose.yml) file.

Check the logs:

```bash
docker-compose logs
```

Stop the container:

```bash
docker-compose down
```

---

# See also 👀

* **[Documentation to run in development 📝](docs/dev)**
* **[Documentation generated from the source code 📖](docs)**
* **[Code of Conduct 🤼](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/CODE_OF_CONDUCT.md)**