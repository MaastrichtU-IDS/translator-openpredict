[![Test production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests-prod.yml) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/run-tests.yml) [![CodeQL analysis](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/codeql-analysis.yml)

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=coverage)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![SonarCloud Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict)

**OpenPredict** is a Python library and API to train and serve predicted biomedical entities associations (e.g. disease treated by drug).

Metadata about runs, models evaluations, features are stored using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html) as RDF.

Access the **Translator OpenPredict API** at **[https://openpredict.semanticscience.org 🔮🐍](https://openpredict.semanticscience.org)**

> You can use this API to retrieve predictions for drug/disease, or add new embeddings to improve the model.

# Deploy the OpenPredict API locally :woman_technologist:

> Requirements: Python 3.6+ and `pip` installed

You can install the `openpredict` python package with `pip` to run the OpenPredict API on your machine, to test new embeddings or improve the library.

We currently recommend to install from the source code `master` branch to get the latest version of OpenPredict. But we also regularly publish the `openpredict` package to PyPI: https://pypi.org/project/openpredict

### With Docker from the source code :whale:

Clone the repository:

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

Start the API in development mode on http://localhost:8808:

```bash
docker-compose up
```

By default all data are stored in the `data/` folder in the directory were you used the `openpredict` command (RDF metadata, features and models of each run)

> Contributions are welcome! If you wish to help improve OpenPredict, see the [instructions to contribute :woman_technologist:](/CONTRIBUTING.md)

You can use the `openpredict` command in the docker container, for example to re-train the baseline model:

```bash
docker-compose exec api openpredict train-model --model openpredict-baseline-omim-drugbank
```

### Reset your local OpenPredict data :wastebasket:

You can easily reset the data of your local OpenPredict deployment by deleting the `data/` folder and restarting the OpenPredict API:

```bash
rm -rf data/
```

> If you are working on improving OpenPredict, you can explore [additional documentation to deploy the OpenPredict API](https://github.com/MaastrichtU-IDS/translator-openpredict/tree/master/docs) locally or with Docker.

### Deploy in production

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

### Test the OpenPredict API

See the [`TESTING.md`](/TESTING.md) file for more details on testing the API.

---

# Use the API​ :mailbox_with_mail:


The user provides a drug or a disease identifier as a CURIE (e.g. DRUGBANK:DB00394, or OMIM:246300), and choose a prediction model (only the `Predict OMIM-DrugBank` classifier is currently implemented).

The API will return predicted targets for the given drug or disease:

* The **potential drugs treating a given disease** :pill:
* The **potential diseases a given drug could treat** :microbe:

> Feel free to try the API at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**

## TRAPI operations

Operations to query OpenPredict using the [Translator Reasoner API](https://github.com/NCATSTranslator/ReasonerAPI) standards.

### Query operation

The `/query` operation will return the same predictions as the `/predict` operation, using the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) format, used within the [Translator project](https://ncats.nih.gov/translator/about).

The user sends a [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) query asking for the predicted targets given: a source, and the relation to predict. The query is a graph with nodes and edges defined in JSON, and uses classes from the [BioLink model](https://biolink.github.io/biolink-model).

You can use the default TRAPI query of OpenPredict `/query` operation to try a working example.

Example of TRAPI query to retrieve drugs similar to a specific drug:

```json
{
    "message": {
        "query_graph": {
        "edges": {
            "e01": {
            "object": "n1",
            "predicates": [
                "biolink:similar_to"
            ],
            "subject": "n0"
            }
        },
        "nodes": {
            "n0": {
            "categories": [
                "biolink:Drug"
            ],
            "ids": [
                "DRUGBANK:DB00394"
            ]
            },
            "n1": {
            "categories": [
                "biolink:Drug"
            ]
            }
        }
        }
    },
    "query_options": {
        "n_results": 3
    }
}
```

### Predicates operation

The `/predicates` operation will return the entities and relations provided by this API in a JSON object (following the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) specifications).

> Try it at [https://openpredict.semanticscience.org/predicates](https://openpredict.semanticscience.org/predicates)

### Notebooks examples :notebook_with_decorative_cover:

We provide [Jupyter Notebooks](https://jupyter.org/) with examples to use the OpenPredict API:

1. [Query the OpenPredict API](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)
2. [Generate embeddings with pyRDF2Vec](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-pyrdf2vec-embeddings.ipynb), and import them in the OpenPredict API

### Add embedding :station:

The default baseline model is `openpredict-baseline-omim-drugbank`. You can choose the base model when you post a new embeddings using the `/embeddings` call. Then the OpenPredict API will:

1. add embeddings to the provided model
2. train the model with the new embeddings
3. store the features and model using a unique ID for the run (e.g. `7621843c-1f5f-11eb-85ae-48a472db7414`)

Once the embedding has been added you can find the existing models previously generated (including `openpredict-baseline-omim-drugbank`), and use them as base model when you ask the model for prediction or add new embeddings.

### Predict operation :crystal_ball:

Use this operation if you just want to easily retrieve predictions for a given entity. The `/predict` operation takes 4 parameters (1 required):

* A `drug_id` to get predicted diseases it could treat (e.g. `DRUGBANK:DB00394`)
  * **OR** a `disease_id` to get predicted drugs it could be treated with (e.g. `OMIM:246300`)
* The prediction model to use (default to `Predict OMIM-DrugBank`)
* The minimum score of the returned predictions, from 0 to 1 (optional)
* The limit of results to return, starting from the higher score, e.g. 42 (optional)

The API will return the list of predicted target for the given entity, the labels are resolved using the [Translator Name Resolver API](https://nodenormalization-sri.renci.org)

> Try it at [https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394](https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394)

---

# More about the data model :minidisc:

* The gold standard for drug-disease indications has been retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979
* Metadata about runs, models evaluations, features are stored as RDF using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html).
  * See the [ML Schema documentation](http://ml-schema.github.io/documentation/ML%20Schema.html) for more details on the data model.

Diagram of the data model used for OpenPredict, based on the ML Schema ontology (`mls`):

![OpenPredict datamodel](https://raw.githubusercontent.com/MaastrichtU-IDS/translator-openpredict/master/docs/OpenPREDICT_datamodel.jpg)

---

# Translator application

### Service Summary
Query for drug-disease pairs predicted from pre-computed sets of graphs embeddings.

Add new embeddings to improve the predictive models, with versioning and scoring of the models.

### Component List
**API component**

1. Component Name: **OpenPredict API**

2. Component Description: **Python API to serve pre-computed set of drug-disease pair predictions from graphs embeddings**

3. GitHub Repository URL: https://github.com/MaastrichtU-IDS/translator-openpredict

4. Component Framework: **Knowledge Provider**

5. System requirements

    5.1. Specific OS and version if required: **python 3.8**

    5.2. CPU/Memory (for CI, TEST and PROD):  **32 CPUs and 32 Go memory ?**

    5.3. Disk size/IO throughput (for CI, TEST and PROD): **20 Go ?**

    5.4. Firewall policies: does the team need access to infrastructure components?
    **The NodeNormalization API https://nodenormalization-sri.renci.org**


6. External Dependencies (any components other than current one)

    6.1. External storage solution: **Models and database are stored in `/data/openpredict` in the Docker container**

7. Docker application:

    7.1. Path to the Dockerfile: **`Dockerfile`**

    7.2. Docker build command:

    ```bash
    docker build ghcr.io/maastrichtu-ids/openpredict-api .
    ```

    7.3. Docker run command:

	**Replace `${PERSISTENT_STORAGE}` with the path to persistent storage on host:**

    ```bash
    docker run -d -v ${PERSISTENT_STORAGE}:/data/openpredict -p 8808:8808 ghcr.io/maastrichtu-ids/openpredict-api
    ```

9. Logs of the application

    9.2. Format of the logs: **TODO**

# Acknowledgments​

* This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
* Predictions made using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/).
* Service funded by the [NIH NCATS Translator project](https://ncats.nih.gov/translator/about).

![Funded the the NIH NCATS Translator project](https://ncats.nih.gov/files/TranslatorGraphic2020_1100x420.jpg)
