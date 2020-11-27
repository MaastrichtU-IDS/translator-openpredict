[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) [![SonarCloud Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=alert_status)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

**OpenPredict** is a Python library and API to train and serve predicted biomedical entities associations (e.g. disease treated by drug). 

Metadata about runs, models evaluations, features are stored using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html) in a RDF triplestore (Ontotext GraphDB).

# Deploy the OpenPredict API locally

Requirements:

* [Docker](https://docs.docker.com/get-docker/) installed (and `docker-compose`)
* Python 3.6+ installed

> Contributions are welcome! If you wish to help improve OpenPredict, see the [instructions to contribute :woman_technologist:](/CONTRIBUTING.md)

The OpenPredict API store its data using a  RDF triplestore. 

* We use will use the open source Virtuoso triplestore in local development environment
* We also use [Ontotext GraphDB](https://github.com/Ontotext-AD/graphdb-docker) in production at IDS, but you are free to use any other triplestore

### Install from the source code

Clone the repository:

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

Install `openpredict` from the source code, and update the package automatically when the files changes locally 🔃

```bash
pip3 install -e .
```

> On Windows you might need to install [Visual Studio C++ 14 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

#### Optional: isolate with a Virtual Environment

If you face conflicts with already installed packages, then you might want to use a [Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to isolate the installation in the current folder before installing OpenPredict:

```bash
# Create the virtual environment folder in your workspace
python3 -m venv .venv
# Activate it using a script in the created folder
source .venv/bin/activate
```

### Start the OpenPredict API

Start the OpenPredict API and the Virtuoso database locally on http://localhost:8890 using Docker (login: `dba` / `dba`):

```bash
docker-compose up -d --force-recreate
```

Start the API (wait 30s for the database to start):

```bash
openpredict start-api
```

> Stop the Virtuoso container:
>
> ```bash
> docker-compose down
> ```

### Reset your local OpenPredict data

Use the `reset_openpredict.sh` script to delete the folders where the OpenPredict API and Virtuoso data are stored, in `data/virtuoso` and `data/openpredict`

```bash
./reset_openpredict.sh
```

> This command uses `sudo` to be able to delete the `data/virtuoso` folder which has been created by the `docker` user.
>
> On Windows: delete all files in `data` folder, just keep `initial-openpredict-metadata.ttl` 

> See more **[documentation to deploy the OpenPredict API](docs/dev)** locally or with Docker.

# Use the API​


The user provides a drug or a disease identifier as a CURIE (e.g. DRUGBANK:DB00394, or OMIM:246300), and choose a prediction model (only the `Predict OMIM-DrugBank` classifier is currently implemented). 

The API will return predicted targets for the given entity, such as:

* The **potential drugs treating a given disease**
* The **potential diseases a given drug could treat**

> Feel free to try the API at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**

### Add embedding

The default baseline model is `openpredict-baseline-omim-drugbank`. You can choose the base model when you post a new embeddings using the `/embeddings` call. Then the OpenPredict API will:

1. add embeddings to the provided model
2. train the model with the new embeddings
3. store the features and model using a unique ID for the run (e.g. `7621843c-1f5f-11eb-85ae-48a472db7414`)

Once the embedding has been added you can find the existing models previously generated (including `openpredict-baseline-omim-drugbank`), and use them as base model when you ask the model for prediction or add new embeddings.

### Predict operation

Use this operation if you just want to easily retrieve predictions for a given entity. The `/predict` operation takes 4 parameters (1 required):

* A source Drug/Disease identifier as a CURIE, e.g. OMIM:246300 (required)
* The prediction model to use (default to `Predict OMIM-DrugBank`)
* The minimum score of the returned predictions, from 0 to 1 (optional)
* The limit of results to return, starting from the higher score, e.g. 42 (optional)  

The API will return the list of predicted target for the given entity, the labels are resolved using the [Translator Name Resolver API](http://robokop.renci.org:2434/docs#/lookup/lookup_curies_lookup_post):

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

> Try it at [https://openpredict.semanticscience.org/predict?entity=DRUGBANK:DB00394](https://openpredict.semanticscience.org/predict?entity=DRUGBANK:DB00394)

### Query operation

The `/query` operation will return the same predictions as the `/predict` operation, using the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) format, used within the [Translator project](https://ncats.nih.gov/translator/about).

The user sends a [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) query asking for the predicted targets given: a source, and the relation to predict. The query is a graph with nodes and edges defined in JSON, and uses classes from the [BioLink model](https://biolink.github.io/biolink-model).

See this [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) query example:

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

The `/predicates` operation will return the entities and relations provided by this API in a JSON object (following the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) specifications).

> Try it at [https://openpredict.semanticscience.org/predicates](https://openpredict.semanticscience.org/predicates)

### Notebook example

You can find a Jupyter Notebook with [examples to query the OpenPredict API on GitHub](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)

# Acknowledgments

* This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
* Predictions made using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/).
* Service funded by the [NIH NCATS Translator project](https://ncats.nih.gov/translator/about). 

![Funded the the NIH NCATS Translator project](https://ncats.nih.gov/files/TranslatorGraphic2020_1100x420.jpg)

