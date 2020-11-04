[![Version](https://img.shields.io/pypi/v/openpredict)](https://pypi.org/project/openpredict) [![Python versions](https://img.shields.io/pypi/pyversions/openpredict)](https://pypi.org/project/openpredict) [![Run tests](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Run%20tests/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Run+tests%22) [![Publish package](https://github.com/MaastrichtU-IDS/translator-openpredict/workflows/Publish%20package/badge.svg)](https://github.com/MaastrichtU-IDS/translator-openpredict/actions?query=workflow%3A%22Publish+package%22) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=alert_status)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=MaastrichtU-IDS_translator-openpredict&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=MaastrichtU-IDS_translator-openpredict) [![CII Best  Practices](https://bestpractices.coreinfrastructure.org/projects/4382/badge)](https://bestpractices.coreinfrastructure.org/projects/4382)

**OpenPredict** is a Python library and API to train and serve predicted biomedical entities associations (e.g. disease treated by drug). 

Metadata about runs, models evaluations, features are stored using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html) in a RDF triplestore (Ontotext GraphDB).

# Use the API 📬

The user provides a drug 💊 or a disease 🦠 identifier as a CURIE (e.g. DRUGBANK:DB00394, or OMIM:246300), and choose a prediction model (only the `Predict OMIM-DrugBank` classifier is currently implemented). 

The API will return predicted targets for the given entity, such as:

* The **potential drugs treating a given disease**
* The **potential diseases a given drug could treat**

> Feel free to try the API at **[openpredict.137.120.31.102.nip.io](https://openpredict.137.120.31.102.nip.io)**

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

> Try it at [https://openpredict.137.120.31.102.nip.io/predict?entity=DRUGBANK:DB00394](https://openpredict.137.120.31.102.nip.io/predict?entity=DRUGBANK:DB00394)

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

> Try it at [https://openpredict.137.120.31.102.nip.io/predicates](https://openpredict.137.120.31.102.nip.io/predicates)

### Notebook example

You can find a Jupyter Notebook with [examples to query the OpenPredict API on GitHub](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/docs/openpredict-examples.ipynb)

# Deploy OpenPredict API

See the **[documentation to deploy the OpenPredict API](docs/dev)** locally or with Docker.

See the [instructions to contribute 👨‍💻](/CONTRIBUTING.md).

# Acknowledgments

* This service has been built from the [fair-workflows/openpredict](https://github.com/fair-workflows/openpredict) project.
* Predictions made using the [PREDICT method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3159979/).
* Service funded by the [NIH NCATS Translator project](https://ncats.nih.gov/translator/about). 

![Funded the the NIH NCATS Translator project](https://ncats.nih.gov/files/TranslatorGraphic2020_1100x420.jpg)

