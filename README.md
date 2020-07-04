[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22)

**Translator OpenPredict** 🔮🐍 is an API to compute and serve predicted biomedical concepts associations using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/), for the [NCATS Translator project](https://ncats.nih.gov/translator/about). 

This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.

# Install the package 📦

```bash
pip install openpredict
```

> PyPI link : [https://pypi.org/project/openpredict](https://pypi.org/project/openpredict)

> You might want to use a `virtualenv` if you are use to it, but this should not be necessary.

---

# Run the API 🌐

### Run from Python script 🐍

```python
from openpredict import openpredict_api

port = 8808
debug = False
openpredict_api.start_api(port, debug)
```

> Access the Swagger UI at [http://localhost:8808/ui](http://localhost:8808/ui)

> Run by default in production, set `debug = True` to run in development mode. 

### Run from the command line ⌨️

Run in production with [Tornado Web Server 🌪️](https://www.tornadoweb.org/en/stable/)

```bash
openpredict start-api
```

> Access the Swagger UI at [http://localhost:8808/ui](http://localhost:8808/ui)

Provide the port as arguments:

```bash
openpredict start-api --port 8808
```

Show help:

```bash
openpredict --help
```

### Run with docker-compose 🐳

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

---

# Compute the model 🤖

Run the pipeline to compute the model used by the API.

### From a Python script 🐍

```python
from openpredict.compute_similarities import get_drug_disease_similarities

get_drug_disease_similarities()
```

### From the command line ⌨️

```bash
openpredict compute-similarities
```

---

# See also 👀

* Browse the [automatically generated Python documentation](docs/package) in **[docs/ 📖](docs)** 
* Read the [documentation to run in development and contribute](docs/contribute) in **[docs/contribute 📝](docs/contribute)** 