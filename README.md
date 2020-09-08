[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) 

**Translator OpenPredict** 🔮🐍 is an API to compute and serve predicted biomedical concepts associations, for the [NCATS Translator project](https://ncats.nih.gov/translator/about). 

This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.

# Use the API 🌐

The Translator OpenPredict API serves predictions of biomedical concepts associations (e.g. disease treated by drug). Feel free to try it at [openpredict.137.120.31.102.nip.io](https://openpredict.137.120.31.102.nip.io)

You can find a Jupyter Notebook with [examples to query the API on GitHub](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)

# Deploy your API 📦

You can also use our tools to build new classifiers, and deploy your OpenPredict API

### Install OpenPredict

You might want to use a virtual environment for Python 3.7 to isolate the installation:

```bash
# Create the virtual environment in your workspace
python3 -m venv .venv
# Activate it
source .venv/bin/activate
```

Install the latest release published on [PyPI 🏷️](https://pypi.org/project/openpredict):

```bash
pip3 install openpredict
```

> Package on PyPI: [https://pypi.org/project/openpredict](https://pypi.org/project/openpredict)

---

### Build the model 🔨

Run the pipeline to compute the model used by the OpenPredict API.

From a Python script:

```python
from openpredict.openpredict_omim_drugbank import build_drug_disease_classifier

build_drug_disease_classifier()
```

Or using the command line:

```bash
openpredict build-models
```

> Work in progress.

---

### Run the API ⚙️

After installing the `openpredict` package (except for docker).

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

Running using Docker can be convenient if you just want to run the API without installing the packages locally, or run in production alongside other services.

Clone the [repository](https://github.com/MaastrichtU-IDS/translator-openpredict):

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

Start the `openpredict-api` container with [docker-compose 🐳](https://docs.docker.com/compose/)

```bash
docker-compose up
```

> Access the Swagger UI at [http://localhost:8808](http://localhost:8808)

> We use [nginx-proxy](https://github.com/nginx-proxy/nginx-proxy) and [docker-letsencrypt-nginx-proxy-companion](https://github.com/nginx-proxy/docker-letsencrypt-nginx-proxy-companion) as reverse proxy for HTTP and HTTPS in production. You can change the proxy URL and port via environment variables `VIRTUAL_HOST`, `VIRTUAL_PORT` and `LETSENCRYPT_HOST` in the [docker-compose.yml](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docker-compose.yml) file.

Stop the container:

```bash
docker-compose down
```

---

# See also 👀

* **[Documentation to run in development 📝](docs/dev)**
* **[Documentation generated from the source code 📖](docs)**
* **[Code of Conduct 🤼](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/CODE_OF_CONDUCT.md)**