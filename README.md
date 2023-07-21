# 🔮🐍 Translator OpenPredict

[![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict)

<!-- [![Test the production API](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/test-prod.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/test-prod.yml) [![Run integration tests for TRAPI](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/test-integration.yml/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions/workflows/test-integration.yml)
-->

This repository contains the code for the **OpenPredict Translator API** available at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**, which serves a few prediction models developed at the Institute of Data Science.

* various folders for **different prediction models** served by the OpenPredict API are available under `src/`:
  * the OpenPredict drug-disease prediction model in `src/openpredict_model/`
  * a model to compile the evidence path between a drug and a disease explaining the predictions of the OpenPredict model in `src/openpredict_evidence_path/`
  * a prediction model trained from the Drug Repurposing Knowledge Graph (aka. DRKG) in `src/drkg_model/`
* the code for the OpenPredict API endpoints in  `src/trapi/` defines:
  *  a TRAPI endpoint returning predictions for the loaded models

The data used by the models in this repository is versionned using `dvc` in the `data/` folder, and stored **on DagsHub at https://dagshub.com/vemonet/translator-openpredict**

### Deploy the OpenPredict API locally

Requirements: Python 3.8+ and `pip` installed

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


Start the API in development mode with docker on http://localhost:8808, the API will automatically reload when you make changes in the code:

```bash
docker compose up api
```

> Contributions are welcome! If you wish to help improve OpenPredict, see the [instructions to contribute :woman_technologist:](/CONTRIBUTING.md) for more details on the development workflow

### Test the OpenPredict API

Run the tests locally with docker:

```bash
docker compose run tests
```

> See the [`TESTING.md`](/TESTING.md) file for more details on testing the API.

You can change the entrypoint of the test container to run other commands, such as training a model:

```bash
docker compose run --entrypoint "python src/openpredict_model/train.py train-model" tests
# Or with the helper script:
./resources/run.sh python src/openpredict_model/train.py train-model
```

### Use the OpenPredict API


The user provides a drug or a disease identifier as a CURIE (e.g. `DRUGBANK:DB00394`, or `OMIM:246300`), and choose a prediction model (only the `Predict OMIM-DrugBank` classifier is currently implemented).

The API will return predicted targets for the given drug or disease:

* The **potential drugs treating a given disease** :pill:
* The **potential diseases a given drug could treat** :microbe:

> Feel free to try the API at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**

#### TRAPI operations

Operations to query OpenPredict using the [Translator Reasoner API](https://github.com/NCATSTranslator/ReasonerAPI) standards.

##### Query operation

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

##### Predicates operation

The `/predicates` operation will return the entities and relations provided by this API in a JSON object (following the [ReasonerAPI](https://github.com/NCATSTranslator/ReasonerAPI) specifications).

> Try it at [https://openpredict.semanticscience.org/predicates](https://openpredict.semanticscience.org/predicates)

#### Notebooks examples :notebook_with_decorative_cover:

We provide [Jupyter Notebooks](https://jupyter.org/) with examples to use the OpenPredict API:

1. [Query the OpenPredict API](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/resources/openpredict-examples.ipynb)
2. [Generate embeddings with pyRDF2Vec](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/resources/openpredict-pyrdf2vec-embeddings.ipynb), and import them in the OpenPredict API

#### Add embedding :station:

The default baseline model is `openpredict_baseline`. You can choose the base model when you post a new embeddings using the `/embeddings` call. Then the OpenPredict API will:

1. add embeddings to the provided model
2. train the model with the new embeddings
3. store the features and model using a unique ID for the run (e.g. `7621843c-1f5f-11eb-85ae-48a472db7414`)

Once the embedding has been added you can find the existing models previously generated (including `openpredict_baseline`), and use them as base model when you ask the model for prediction or add new embeddings.

#### Predict operation :crystal_ball:

Use this operation if you just want to easily retrieve predictions for a given entity. The `/predict` operation takes 4 parameters (1 required):

* A `drug_id` to get predicted diseases it could treat (e.g. `DRUGBANK:DB00394`)
  * **OR** a `disease_id` to get predicted drugs it could be treated with (e.g. `OMIM:246300`)
* The prediction model to use (default to `Predict OMIM-DrugBank`)
* The minimum score of the returned predictions, from 0 to 1 (optional)
* The limit of results to return, starting from the higher score, e.g. 42 (optional)

The API will return the list of predicted target for the given entity, the labels are resolved using the [Translator Name Resolver API](https://nodenormalization-sri.renci.org)

> Try it at [https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394](https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394)

---

### More about the data model :minidisc:

* The gold standard for drug-disease indications has been retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979
* Metadata about runs, models evaluations, features are stored as RDF using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html).
  * See the [ML Schema documentation](http://ml-schema.github.io/documentation/ML%20Schema.html) for more details on the data model.

Diagram of the data model used for OpenPredict, based on the ML Schema ontology (`mls`):

![OpenPredict datamodel](https://raw.githubusercontent.com/MaastrichtU-IDS/translator-openpredict/master/resources/OpenPREDICT_datamodel.jpg)

---

## Translator application

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
