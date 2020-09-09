[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) 

**Translator OpenPredict** 🔮🐍 is an API to compute and serve predicted biomedical concepts associations, for the [NCATS Translator project](https://ncats.nih.gov/translator/about). 

# Use the API 🌐

The OpenPredict API serves predictions of biomedical concepts associations (e.g. disease treated by drug). 

The user provides a drug 💊 or a disease 🦠 identifier as a CURIE (e.g. DRUGBANK:DB00394, OMIM:246300), and choose a prediction model (only OMIM - DrugBank at the moment). 

The API will return the predicted targets for the given entity:

* The potential drugs treating a given disease
* The potential diseases a given drug could treat

> Feel free to try it at **[openpredict.137.120.31.102.nip.io](https://openpredict.137.120.31.102.nip.io)**

### Predict operation

The `/predict` operation takes 4 parameters:

* A source Drug/Disease identifier as a CURIE
* The prediction model to use
* The minimum score of the returned predictions (optional)
* The limit of results to return, starting from the higher score (optional)  

It will return the list of predicted target for the given entity, the labels are resolved using the [Translator Name Resolver API](http://robokop.renci.org:2434/docs#/lookup/lookup_curies_lookup_post):

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

### Query operation

The `/query` operation will return the same predictions in the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) format, used within the [Translator project](https://ncats.nih.gov/translator/about).

The user sends a ReasonerAPI query asking for the predicted target nodes given a source node and the relation to predict between those 2 nodes. See this example of a ReasonerAPI query:

```json
{
  "message": {
    "query_graph": {
      "edges": [
        {
          "id": "e00",
          "source_id": "n00",
          "target_id": "n01",
          "type": "treated_by"
        }
      ],
      "nodes": [
        {
          "curie": "DRUGBANK:DB00394",
          "id": "n00",
          "type": "drug"
        },
        {
          "id": "n01",
          "type": "disease"
        }
      ]
    }
  }
}
```

### Predicates operation

The `/predicates` operation will return the entities and relations returned by this API (following the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) specifications).

### Notebook example

You can find a Jupyter Notebook with [examples to query the API on GitHub](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)

# Deploy your API 📦

You can also use OpenPredict to build new classifiers, and deploy your API.

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

openpredict_api.start_api(8808)
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

## Acknowledgments

* This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
* Predictions made using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/)

# See also 👀

* **[Documentation to install and run in development 📝](docs/dev)**
* **[Documentation generated from the source code 📖](docs)**