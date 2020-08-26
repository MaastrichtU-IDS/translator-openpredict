# Table of Contents

* [openpredict](#.openpredict)
* [openpredict.openpredict\_omim\_drugbank](#.openpredict.openpredict_omim_drugbank)
  * [adjcencydict2matrix](#.openpredict.openpredict_omim_drugbank.adjcencydict2matrix)
  * [mergeFeatureMatrix](#.openpredict.openpredict_omim_drugbank.mergeFeatureMatrix)
  * [generatePairs](#.openpredict.openpredict_omim_drugbank.generatePairs)
  * [balance\_data](#.openpredict.openpredict_omim_drugbank.balance_data)
  * [geometricMean](#.openpredict.openpredict_omim_drugbank.geometricMean)
  * [createFeatureArray](#.openpredict.openpredict_omim_drugbank.createFeatureArray)
  * [createFeatureDF](#.openpredict.openpredict_omim_drugbank.createFeatureDF)
  * [calculateCombinedSimilarity](#.openpredict.openpredict_omim_drugbank.calculateCombinedSimilarity)
  * [trainModel](#.openpredict.openpredict_omim_drugbank.trainModel)
  * [multimetric\_score](#.openpredict.openpredict_omim_drugbank.multimetric_score)
  * [evaluate](#.openpredict.openpredict_omim_drugbank.evaluate)
  * [get\_drug\_disease\_classifier](#.openpredict.openpredict_omim_drugbank.get_drug_disease_classifier)
  * [query\_omim\_drugbank\_classifier](#.openpredict.openpredict_omim_drugbank.query_omim_drugbank_classifier)
* [openpredict.openpredict\_api](#.openpredict.openpredict_api)
  * [start\_api](#.openpredict.openpredict_api.start_api)
  * [get\_predict](#.openpredict.openpredict_api.get_predict)
  * [post\_reasoner\_predict](#.openpredict.openpredict_api.post_reasoner_predict)
* [openpredict.utils](#.openpredict.utils)
* [openpredict.\_\_main\_\_](#.openpredict.__main__)
  * [main](#.openpredict.__main__.main)

<a name=".openpredict"></a>
# openpredict

<a name=".openpredict.openpredict_omim_drugbank"></a>
# openpredict.openpredict\_omim\_drugbank

<a name=".openpredict.openpredict_omim_drugbank.adjcencydict2matrix"></a>
#### adjcencydict2matrix

```python
adjcencydict2matrix(df, name1, name2)
```

Convert dict to matrix

**Arguments**:

- `df`: Dataframe
- `name1`: index name
- `name2`: columns name

<a name=".openpredict.openpredict_omim_drugbank.mergeFeatureMatrix"></a>
#### mergeFeatureMatrix

```python
mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
```

Merge the drug and disease feature matrix

**Arguments**:

- `drugfeatfiles`: Drug features files list
- `diseasefeatfiles`: Disease features files list

<a name=".openpredict.openpredict_omim_drugbank.generatePairs"></a>
#### generatePairs

```python
generatePairs(drug_df, disease_df, drugDiseaseKnown)
```

Generate positive and negative pairs using the Drug dataframe, the Disease dataframe and known drug-disease associations dataframe

**Arguments**:

- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `drugDiseaseKnown`: Known drug-disease association dataframe

<a name=".openpredict.openpredict_omim_drugbank.balance_data"></a>
#### balance\_data

```python
balance_data(pairs, classes, n_proportion)
```

Balance negative and positives samples

**Arguments**:

- `pairs`: Positive/negative pairs previously generated
- `classes`: Classes corresponding to the pairs
- `n_proportion`: Proportion number, e.g. 2

<a name=".openpredict.openpredict_omim_drugbank.geometricMean"></a>
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

<a name=".openpredict.openpredict_omim_drugbank.createFeatureArray"></a>
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

<a name=".openpredict.openpredict_omim_drugbank.createFeatureDF"></a>
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

<a name=".openpredict.openpredict_omim_drugbank.calculateCombinedSimilarity"></a>
#### calculateCombinedSimilarity

```python
calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease)
```

Compute combined similarities

**Arguments**:

- `pairs_train`: Pairs used to train
- `pairs_test`: Pairs used to test
- `classes_train`: Classes corresponding to the pairs used to train
- `classes_test`: Classes corresponding to the pairs used to test
- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `knownDrugDisease`: Known drug-disease associations

<a name=".openpredict.openpredict_omim_drugbank.trainModel"></a>
#### trainModel

```python
trainModel(train_df, clf)
```

Train model

**Arguments**:

- `train_df`: Train dataframe
- `clf`: Classifier

<a name=".openpredict.openpredict_omim_drugbank.multimetric_score"></a>
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

<a name=".openpredict.openpredict_omim_drugbank.evaluate"></a>
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

<a name=".openpredict.openpredict_omim_drugbank.get_drug_disease_classifier"></a>
#### get\_drug\_disease\_classifier

```python
get_drug_disease_classifier()
```

The main function to run the drug-disease similarities pipeline,
and build the drug-disease classifier.
It returns, and stores the generated classifier as a `.joblib` file
in the `data/models` folder,

**Returns**:

Classifier of predicted similarities and scores

<a name=".openpredict.openpredict_omim_drugbank.query_omim_drugbank_classifier"></a>
#### query\_omim\_drugbank\_classifier

```python
query_omim_drugbank_classifier(input_curie)
```

The main function to query the drug-disease OpenPredict classifier,
It queries the previously generated classifier a `.joblib` file
in the `data/models` folder

**Returns**:

Predictions and scores

<a name=".openpredict.openpredict_api"></a>
# openpredict.openpredict\_api

<a name=".openpredict.openpredict_api.start_api"></a>
#### start\_api

```python
start_api(port=8808, debug=False, start_spark=True)
```

Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

**Arguments**:

- `port`: Port of the OpenPredict API, defaults to 8808
- `debug`: Run in debug mode, defaults to False
- `start_spark`: Start a local Spark cluster, default to true

<a name=".openpredict.openpredict_api.get_predict"></a>
#### get\_predict

```python
get_predict(entity, classifier="OpenPredict OMIM-DrugBank")
```

Get predicted associations for a given entity CURIE.

**Arguments**:

- `entity`: Search for predicted associations for this entity CURIE

**Returns**:

Prediction results object with score

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

<a name=".openpredict.utils"></a>
# openpredict.utils

<a name=".openpredict.__main__"></a>
# openpredict.\_\_main\_\_

<a name=".openpredict.__main__.main"></a>
#### main

```python
@click.group()
main(args=None)
```

Command Line Interface to run OpenPredict

