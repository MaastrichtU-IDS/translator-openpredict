# Table of Contents

* [openpredict](#.openpredict)
* [openpredict.compute\_similarities](#.openpredict.build_models)
  * [adjcencydict2matrix](#.openpredict.build_models.adjcencydict2matrix)
  * [mergeFeatureMatrix](#.openpredict.build_models.mergeFeatureMatrix)
  * [generatePairs](#.openpredict.build_models.generatePairs)
  * [balance\_data](#.openpredict.build_models.balance_data)
  * [geometricMean](#.openpredict.build_models.geometricMean)
  * [createFeatureDF](#.openpredict.build_models.createFeatureDF)
  * [calculateCombinedSimilarity](#.openpredict.build_models.calculateCombinedSimilarity)
  * [trainModel](#.openpredict.build_models.trainModel)
  * [multimetric\_score](#.openpredict.build_models.multimetric_score)
  * [evaluate](#.openpredict.build_models.evaluate)
  * [get\_drug\_disease\_classifier](#.openpredict.build_models.get_drug_disease_classifier)
* [openpredict.openpredict\_api](#.openpredict.openpredict_api)
  * [start\_api](#.openpredict.openpredict_api.start_api)
  * [get\_predict](#.openpredict.openpredict_api.get_predict)
  * [post\_reasoner\_predict](#.openpredict.openpredict_api.post_reasoner_predict)
* [openpredict.\_\_main\_\_](#.openpredict.__main__)
  * [main](#.openpredict.__main__.main)

<a name=".openpredict"></a>
# openpredict

<a name=".openpredict.build_models"></a>
# openpredict.compute\_similarities

<a name=".openpredict.build_models.adjcencydict2matrix"></a>
#### adjcencydict2matrix

```python
adjcencydict2matrix(df, name1, name2)
```

Convert dict to matrix

**Arguments**:

- `df`: Dataframe
- `name1`: index name
- `name2`: columns name

<a name=".openpredict.build_models.mergeFeatureMatrix"></a>
#### mergeFeatureMatrix

```python
mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
```

Merge the drug and disease feature matrix

**Arguments**:

- `drugfeatfiles`: Drug features files list
- `diseasefeatfiles`: Disease features files list

<a name=".openpredict.build_models.generatePairs"></a>
#### generatePairs

```python
generatePairs(drug_df, disease_df, drugDiseaseKnown)
```

Generate positive and negative pairs using the Drug dataframe, the Disease dataframe and known drug-disease associations dataframe

**Arguments**:

- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `drugDiseaseKnown`: Known drug-disease association dataframe

<a name=".openpredict.build_models.balance_data"></a>
#### balance\_data

```python
balance_data(pairs, classes, n_proportion)
```

Balance negative and positives samples

**Arguments**:

- `pairs`: Positive/negative pairs previously generated
- `classes`: Classes corresponding to the pairs
- `n_proportion`: Proportion number, e.g. 2

<a name=".openpredict.build_models.geometricMean"></a>
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

<a name=".openpredict.build_models.createFeatureDF"></a>
#### createFeatureDF

```python
createFeatureDF(pairs, classes, knownDrugDisease, drugDFs, diseaseDFs)
```

Create the features dataframes

**Arguments**:

- `pairs`: Generated pairs
- `classes`: Classes corresponding to the pairs
- `knownDrugDisease`: Known drug-disease associations
- `drugDFs`: Drug dataframes
- `diseaseDFs`: Disease dataframes

**Returns**:

The features dataframe

<a name=".openpredict.build_models.calculateCombinedSimilarity"></a>
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

<a name=".openpredict.build_models.trainModel"></a>
#### trainModel

```python
trainModel(train_df, clf)
```

Train model

**Arguments**:

- `train_df`: Train dataframe
- `clf`: Classifier

<a name=".openpredict.build_models.multimetric_score"></a>
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

<a name=".openpredict.build_models.evaluate"></a>
#### evaluate

```python
evaluate(train_df, test_df, clf)
```

Evaluate the trained classifier

**Arguments**:

- `train_df`: Train dataframe
- `test_df`: Test dataframe
- `clf`: Classifier

**Returns**:

Scores

<a name=".openpredict.build_models.get_drug_disease_classifier"></a>
#### get\_drug\_disease\_classifier

```python
get_drug_disease_classifier()
```

The main function to run the drug-disease similarities pipeline,
and build the drug-disease classifier.
It returns, and stores the generated classifier as a `.joblib` file
in the `data/models` folder,

**Returns**:

Classifier of predicted similarities

<a name=".openpredict.openpredict_api"></a>
# openpredict.openpredict\_api

<a name=".openpredict.openpredict_api.start_api"></a>
#### start\_api

```python
start_api(port=8808, debug=False)
```

Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

**Arguments**:

- `port`: Port of the OpenPredict API, defaults to 8808
- `debug`: Run in debug mode, defaults to False

<a name=".openpredict.openpredict_api.get_predict"></a>
#### get\_predict

```python
get_predict(entity, input_type, predict_type)
```

Get predicted associations for a given entity.

**Arguments**:

- `entity`: Search for predicted associations for this entity
- `input_type`: Type of the entity in the input (e.g. drug, disease)
- `predict_type`: Type of the predicted entity in the output (e.g. drug, disease)

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

<a name=".openpredict.__main__"></a>
# openpredict.\_\_main\_\_

<a name=".openpredict.__main__.main"></a>
#### main

```python
@click.group()
main(args=None)
```

Command Line Interface to run OpenPredict

