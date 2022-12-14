# Table of Contents

* [openpredict](#.openpredict)
* [openpredict.reasonerapi\_parser](#.openpredict.trapi_parser)
  * [typed\_results\_to\_reasonerapi](#.openpredict.trapi_parser.resolve_trapi_query)
* [openpredict.rdf\_utils](#.openpredict.rdf_utils)
  * [insert\_graph\_in\_sparql\_endpoint](#.openpredict.rdf_utils.insert_graph_in_sparql_endpoint)
  * [query\_sparql\_endpoint](#.openpredict.rdf_utils.query_sparql_endpoint)
  * [init\_triplestore](#.openpredict.rdf_utils.init_triplestore)
  * [add\_feature\_metadata](#.openpredict.rdf_utils.add_feature_metadata)
  * [add\_run\_metadata](#.openpredict.rdf_utils.add_run_metadata)
  * [retrieve\_features](#.openpredict.rdf_utils.retrieve_features)
  * [retrieve\_models](#.openpredict.rdf_utils.retrieve_models)
* [openpredict.openpredict\_utils](#.openpredict.utils)
  * [get\_openpredict\_dir](#.openpredict.utils.get_openpredict_dir)
  * [init\_openpredict\_dir](#.openpredict.utils.init_openpredict_dir)
  * [get\_entities\_labels](#.openpredict.utils.get_entities_labels)
* [openpredict.openpredict\_api](#.openpredict.openpredict_api)
  * [start\_api](#.openpredict.openpredict_api.start_api)
  * [post\_embedding](#.openpredict.openpredict_api.post_embedding)
  * [get\_predict](#.openpredict.openpredict_api.get_predict)
  * [get\_predicates](#.openpredict.openpredict_api.get_predicates)
  * [get\_features](#.openpredict.openpredict_api.get_features)
  * [get\_models](#.openpredict.openpredict_api.get_models)
  * [post\_reasoner\_predict](#.openpredict.openpredict_api.post_reasoner_predict)
* [openpredict.\_\_main\_\_](#.openpredict.__main__)
  * [main](#.openpredict.__main__.main)
* [openpredict.openpredict\_model](#.openpredict.models.openpredict_model)
  * [get\_spark\_context](#.openpredict.models.openpredict_model.get_spark_context)
  * [adjcencydict2matrix](#.openpredict.models.openpredict_model.adjcencydict2matrix)
  * [addEmbedding](#.openpredict.models.openpredict_model.addEmbedding)
  * [mergeFeatureMatrix](#.openpredict.models.openpredict_model.mergeFeatureMatrix)
  * [generatePairs](#.openpredict.models.openpredict_model.generatePairs)
  * [balance\_data](#.openpredict.models.openpredict_model.balance_data)
  * [geometricMean](#.openpredict.models.openpredict_model.geometricMean)
  * [createFeatureArray](#.openpredict.models.openpredict_model.createFeatureArray)
  * [sparkBuildFeatures](#.openpredict.models.openpredict_model.sparkBuildFeatures)
  * [createFeatureDF](#.openpredict.models.openpredict_model.createFeatureDF)
  * [calculateCombinedSimilarity](#.openpredict.models.openpredict_model.calculateCombinedSimilarity)
  * [train\_classifier](#.openpredict.models.openpredict_model.train_classifier)
  * [multimetric\_score](#.openpredict.models.openpredict_model.multimetric_score)
  * [evaluate](#.openpredict.models.openpredict_model.evaluate)
  * [createFeaturesSparkOrDF](#.openpredict.models.openpredict_model.createFeaturesSparkOrDF)
  * [train\_model](#.openpredict.models.openpredict_model.train_model)
  * [query\_omim\_drugbank\_classifier](#.openpredict.models.openpredict_model.query_omim_drugbank_classifier)
  * [get\_predictions](#.openpredict.models.openpredict_model.get_predictions)

<a name=".openpredict"></a>
# openpredict

<a name=".openpredict.trapi_parser"></a>
# openpredict.reasonerapi\_parser

<a name=".openpredict.trapi_parser.resolve_trapi_query"></a>
#### typed\_results\_to\_reasonerapi

```python
resolve_trapi_query(reasoner_query, model_id)
```

Convert an array of predictions objects to ReasonerAPI format
Run the get_predict to get the QueryGraph edges and nodes
{disease: OMIM:1567, drug: DRUGBANK:DB0001, score: 0.9}

:param: reasoner_query Query from Reasoner API

**Returns**:

Results as ReasonerAPI object

<a name=".openpredict.rdf_utils"></a>
# openpredict.rdf\_utils

<a name=".openpredict.rdf_utils.insert_graph_in_sparql_endpoint"></a>
#### insert\_graph\_in\_sparql\_endpoint

```python
insert_graph_in_sparql_endpoint(g)
```

Insert rdflib graph in a Update SPARQL endpoint using SPARQLWrapper

**Arguments**:

- `g`: rdflib graph to insert

**Returns**:

SPARQL update query result

<a name=".openpredict.rdf_utils.query_sparql_endpoint"></a>
#### query\_sparql\_endpoint

```python
query_sparql_endpoint(query)
```

Run select SPARQL query against SPARQL endpoint

**Arguments**:

- `query`: SPARQL query as a string

**Returns**:

Object containing the result bindings

<a name=".openpredict.rdf_utils.init_triplestore"></a>
#### init\_triplestore

```python
init_triplestore()
```

Only initialized the triplestore if no run for openpredict_baseline can be found.
Init using the data/openpredict-metadata.ttl RDF file

<a name=".openpredict.rdf_utils.add_feature_metadata"></a>
#### add\_feature\_metadata

```python
add_feature_metadata(id, description, type)
```

Generate RDF metadata for a feature

**Arguments**:

- `id`: if used to identify the feature
- `description`: feature description
- `type`: feature type

**Returns**:

rdflib graph after loading the feature

<a name=".openpredict.rdf_utils.add_run_metadata"></a>
#### add\_run\_metadata

```python
add_run_metadata(scores, model_features, hyper_params)
```

Generate RDF metadata for a classifier and save it in data/openpredict-metadata.ttl, based on OpenPredict model:
https://github.com/fair-workflows/openpredict/blob/master/data/rdf/results_disjoint_lr.nq

**Arguments**:

- `scores`: scores
- `model_features`: List of features in the model
- `label`: label of the classifier

**Returns**:

Run id

<a name=".openpredict.rdf_utils.retrieve_features"></a>
#### retrieve\_features

```python
retrieve_features(type='All')
```

Get features in the ML model

**Arguments**:

- `type`: type of the feature (All, Both, Drug, Disease)

**Returns**:

JSON with features

<a name=".openpredict.rdf_utils.retrieve_models"></a>
#### retrieve\_models

```python
retrieve_models()
```

Get models with their scores and features

**Returns**:

JSON with models and features

<a name=".openpredict.utils"></a>
# openpredict.openpredict\_utils

<a name=".openpredict.utils.get_openpredict_dir"></a>
#### get\_openpredict\_dir

```python
get_openpredict_dir(subfolder='')
```

Return the full path to the provided files in the OpenPredict data folder
Where models and features for runs are stored

<a name=".openpredict.utils.init_openpredict_dir"></a>
#### init\_openpredict\_dir

```python
init_openpredict_dir()
```

Create OpenPredict folder and initiatite files if necessary.
Also create baseline features in the triplestore

<a name=".openpredict.utils.get_entities_labels"></a>
#### get\_entities\_labels

```python
get_entities_labels(entity_list)
```

Send the list of node IDs to Translator Normalization API to get labels
See API: https://nodenormalization-sri.renci.org/apidocs/#/Interfaces/get_get_normalized_nodes
and example notebook: https://github.com/TranslatorIIPrototypes/NodeNormalization/blob/master/documentation/NodeNormalization.ipynb

<a name=".openpredict.openpredict_api"></a>
# openpredict.openpredict\_api

<a name=".openpredict.openpredict_api.start_api"></a>
#### start\_api

```python
start_api(port=8808, debug=False, start_spark=True)
```

Start the Translator OpenPredict API

**Arguments**:

- `port`: Port of the OpenPredict API, defaults to 8808
- `debug`: Run in debug mode, defaults to False
- `start_spark`: Start a local Spark cluster, default to true

<a name=".openpredict.openpredict_api.post_embedding"></a>
#### post\_embedding

```python
post_embedding(types, emb_name, description, model_id)
```

Post JSON embeddings via the API, with simple APIKEY authentication
provided in environment variables

<a name=".openpredict.openpredict_api.get_predict"></a>
#### get\_predict

```python
get_predict(entity, model_id, score=None, n_results=None)
```

Get predicted associations for a given entity CURIE.

**Arguments**:

- `entity`: Search for predicted associations for this entity CURIE

**Returns**:

Prediction results object with score

<a name=".openpredict.openpredict_api.get_predicates"></a>
#### get\_predicates

```python
get_predicates()
```

Get predicates and entities provided by the API

**Returns**:

JSON with biolink entities

<a name=".openpredict.openpredict_api.get_features"></a>
#### get\_features

```python
get_features(type)
```

Get features in the model

**Returns**:

JSON with features

<a name=".openpredict.openpredict_api.get_models"></a>
#### get\_models

```python
get_models()
```

Get models with their scores and features

**Returns**:

JSON with models and features

<a name=".openpredict.openpredict_api.post_reasoner_predict"></a>
#### post\_reasoner\_predict

```python
post_reasoner_predict(request_body)
```

Get predicted associations for a given ReasonerAPI query.

**Arguments**:

- `request_body`: The ReasonerStdAPI query in JSON

**Returns**:

Predictions as a ReasonerStdAPI Message

<a name=".openpredict.__main__"></a>
# openpredict.\_\_main\_\_

<a name=".openpredict.__main__.main"></a>
#### main

```python
@click.group()
main(args=None)
```

Command Line Interface to run OpenPredict

<a name=".openpredict.models.openpredict_model"></a>
# openpredict.openpredict\_model

<a name=".openpredict.models.openpredict_model.get_spark_context"></a>
#### get\_spark\_context

```python
get_spark_context()
```

Get Spark context, either from Spark Master URL to a Spark cluster
If not URL is provided, then it will try to run Spark locally

**Returns**:

Spark context

<a name=".openpredict.models.openpredict_model.adjcencydict2matrix"></a>
#### adjcencydict2matrix

```python
adjcencydict2matrix(df, name1, name2)
```

Convert dict to matrix

**Arguments**:

- `df`: Dataframe
- `name1`: index name
- `name2`: columns name

<a name=".openpredict.models.openpredict_model.addEmbedding"></a>
#### addEmbedding

```python
run_id, scores = addEmbedding(embedding_file, emb_name, types, description, from_model_id)
```

Add embedding to the drug similarity matrix dataframe

**Arguments**:

- `embedding_file`: JSON file containing records ('entity': id, 'embdding': array of numbers )
- `emb_name`: new column name to be added
- `types`: types in the embedding vector ['Drugs', 'Diseases', 'Both']
- `description`: description of the embedding provenance

<a name=".openpredict.models.openpredict_model.mergeFeatureMatrix"></a>
#### mergeFeatureMatrix

```python
mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
```

Merge the drug and disease feature matrix

**Arguments**:

- `drugfeatfiles`: Drug features files list
- `diseasefeatfiles`: Disease features files list

<a name=".openpredict.models.openpredict_model.generatePairs"></a>
#### generatePairs

```python
generatePairs(drug_df, disease_df, drugDiseaseKnown)
```

Generate positive and negative pairs using the Drug dataframe,
the Disease dataframe and known drug-disease associations dataframe

**Arguments**:

- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `drugDiseaseKnown`: Known drug-disease association dataframe

<a name=".openpredict.models.openpredict_model.balance_data"></a>
#### balance\_data

```python
balance_data(pairs, classes, n_proportion)
```

Balance negative and positives samples

**Arguments**:

- `pairs`: Positive/negative pairs previously generated
- `classes`: Classes corresponding to the pairs
- `n_proportion`: Proportion number, e.g. 2

<a name=".openpredict.models.openpredict_model.geometricMean"></a>
#### geometricMean

```python
geometricMean(drug, disease, knownDrugDisease, drugDF, diseaseDF)
```

Compute the geometric means of a drug-disease association using previously generated dataframes

**Arguments**:

- `drug`: Drug
- `disease`: Disease
- `knownDrugDisease`: Known drug-disease associations
- `drugDF`: Drug dataframe
- `diseaseDF`: Disease dataframe

<a name=".openpredict.models.openpredict_model.createFeatureArray"></a>
#### createFeatureArray

```python
createFeatureArray(drug, disease, knownDrugDisease, drugDFs, diseaseDFs)
```

Create the features dataframes for Spark.

**Arguments**:

- `drug`: Drug
- `disease`: Disease
- `knownDrugDisease`: Known drug-disease associations
- `drugDFs`: Drug dataframes
- `diseaseDFs`: Disease dataframes

**Returns**:

The features dataframe

<a name=".openpredict.models.openpredict_model.sparkBuildFeatures"></a>
#### sparkBuildFeatures

```python
sparkBuildFeatures(sc, pairs, classes, knownDrugDis, drug_df, disease_df)
```

Create the feature matrix for Spark.

**Arguments**:

- `sc`: Spark context
- `pairs`: Generated pairs
- `classes`: Classes corresponding to the pairs
- `knownDrugDisease`: Known drug-disease associations
- `drugDFs`: Drug dataframes
- `diseaseDFs`: Disease dataframes

**Returns**:

The features dataframe

<a name=".openpredict.models.openpredict_model.createFeatureDF"></a>
#### createFeatureDF

```python
createFeatureDF(pairs, classes, knownDrugDisease, drugDFs, diseaseDFs)
```

Create the features dataframes.

**Arguments**:

- `pairs`: Generated pairs
- `classes`: Classes corresponding to the pairs
- `knownDrugDisease`: Known drug-disease associations
- `drugDFs`: Drug dataframes
- `diseaseDFs`: Disease dataframes

**Returns**:

The features dataframe

<a name=".openpredict.models.openpredict_model.calculateCombinedSimilarity"></a>
#### calculateCombinedSimilarity

```python
calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease)
```

Compute combined similarities. Use Spark if available for speed, otherwise use pandas

**Arguments**:

- `pairs_train`: Pairs used to train
- `pairs_test`: Pairs used to test
- `classes_train`: Classes corresponding to the pairs used to train
- `classes_test`: Classes corresponding to the pairs used to test
- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `knownDrugDisease`: Known drug-disease associations

<a name=".openpredict.models.openpredict_model.train_classifier"></a>
#### train\_classifier

```python
train_classifier(train_df, clf)
```

Train classifier

**Arguments**:

- `train_df`: Train dataframe
- `clf`: Classifier

<a name=".openpredict.models.openpredict_model.multimetric_score"></a>
#### multimetric\_score

```python
multimetric_score(estimator, X_test, y_test, scorers)
```

Return a dict of score for multimetric scoring

**Arguments**:

- `estimator`: Estimator
- `X_test`: X test
- `y_test`: Y test
- `scorers`: Dict of scorers

**Returns**:

Multimetric scores

<a name=".openpredict.models.openpredict_model.evaluate"></a>
#### evaluate

```python
evaluate(test_df, clf)
```

Evaluate the trained classifier

**Arguments**:

- `test_df`: Test dataframe
- `clf`: Classifier

**Returns**:

Scores

<a name=".openpredict.models.openpredict_model.createFeaturesSparkOrDF"></a>
#### createFeaturesSparkOrDF

```python
createFeaturesSparkOrDF(pairs, classes, drug_df, disease_df)
```

Create features dataframes. Use Spark if available for speed, otherwise use pandas

**Arguments**:

- `pairs`: pairs
- `classes`: classes
- `drug_df`: drug
- `disease_df`: disease dataframe

**Returns**:

Feature dataframe

<a name=".openpredict.models.openpredict_model.train_model"></a>
#### train\_model

```python
train_model(from_model_id='openpredict_baseline')
```

The main function to run the drug-disease similarities pipeline,
and train the drug-disease classifier.
It returns, and stores the generated classifier as a `.joblib` file
in the `data/models` folder,

**Arguments**:

- `from_scratch`: Train the model for scratch (True by default)

**Returns**:

Classifier of predicted similarities and scores

<a name=".openpredict.models.openpredict_model.query_omim_drugbank_classifier"></a>
#### query\_omim\_drugbank\_classifier

```python
query_omim_drugbank_classifier(input_curie, model_id)
```

The main function to query the drug-disease OpenPredict classifier,
It queries the previously generated classifier a `.joblib` file
in the `data/models` folder

**Returns**:

Predictions and scores

<a name=".openpredict.models.openpredict_model.get_predictions"></a>
#### get\_predictions

```python
get_predictions(id_to_predict, model_id, min_score=None, max_score=None, n_results=None)
```

Run classifiers to get predictions

**Arguments**:

- `id_to_predict`: Id of the entity to get prediction from
- `classifier`: classifier used to get the predictions
- `score`: score minimum of predictions
- `n_results`: number of predictions to return

**Returns**:

predictions in array of JSON object
