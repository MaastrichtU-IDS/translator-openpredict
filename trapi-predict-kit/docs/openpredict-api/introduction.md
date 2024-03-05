An example of KP build with the `trapi-predict-kit` is **OpenPredict** available at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**, for drug-treats-disease associations predicted using the OpenPredict model.

The data used by the models in this repository is versionned using `dvc` in the `data/` folder, and stored **on DagsHub at [dagshub.com/vemonet/translator-openpredict](https://dagshub.com/vemonet/translator-openpredict)**


### Use the OpenPredict API


The user provides a drug or a disease identifier as a CURIE (e.g. DRUGBANK:DB00394, or OMIM:246300), and choose a prediction model (only the `Predict OMIM-DrugBank` classifier is currently implemented).

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

#### Notebooks examples

We provide [Jupyter Notebooks](https://jupyter.org/) with examples to use the OpenPredict API:

1. [Query the OpenPredict API](https://github.com/MaastrichtU-IDS/trapi-predict-kit/blob/master/docs/openpredict-examples.ipynb)
2. [Generate embeddings with pyRDF2Vec](https://github.com/MaastrichtU-IDS/trapi-predict-kit/blob/master/docs/openpredict-pyrdf2vec-embeddings.ipynb), and import them in the OpenPredict API

#### Add embedding

The default baseline model is `openpredict_baseline`. You can choose the base model when you post a new embeddings using the `/embeddings` call. Then the OpenPredict API will:

1. add embeddings to the provided model
2. train the model with the new embeddings
3. store the features and model using a unique ID for the run (e.g. `7621843c-1f5f-11eb-85ae-48a472db7414`)

Once the embedding has been added you can find the existing models previously generated (including `openpredict_baseline`), and use them as base model when you ask the model for prediction or add new embeddings.

#### Predict operation

Use this operation if you just want to easily retrieve predictions for a given entity. The `/predict` operation takes 4 parameters (1 required):

* A `drug_id` to get predicted diseases it could treat (e.g. `DRUGBANK:DB00394`)
  * **OR** a `disease_id` to get predicted drugs it could be treated with (e.g. `OMIM:246300`)
* The prediction model to use (default to `Predict OMIM-DrugBank`)
* The minimum score of the returned predictions, from 0 to 1 (optional)
* The limit of results to return, starting from the higher score, e.g. 42 (optional)

The API will return the list of predicted target for the given entity, the labels are resolved using the [Translator Name Resolver API](https://nodenormalization-sri.renci.org)

> Try it at [https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394](https://openpredict.semanticscience.org/predict?drug_id=DRUGBANK:DB00394)

---

### More about the data model

* The gold standard for drug-disease indications has been retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979
* Metadata about runs, models evaluations, features are stored as RDF using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html).
  * See the [ML Schema documentation](http://ml-schema.github.io/documentation/ML%20Schema.html) for more details on the data model.

Diagram of the data model used for OpenPredict, based on the ML Schema ontology (`mls`):

![OpenPredict datamodel](https://raw.githubusercontent.com/MaastrichtU-IDS/trapi-predict-kit/master/docs/OpenPREDICT_datamodel.jpg)
