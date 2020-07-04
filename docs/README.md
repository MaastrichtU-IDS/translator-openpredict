# OpenPredict Package documentation 🔮🐍

* [openpredict](#.openpredict)
* [openpredict.compute\_similarities](#.openpredict.compute_similarities)
  * [adjcencydict2matrix](#.openpredict.compute_similarities.adjcencydict2matrix)
  * [mergeFeatureMatrix](#.openpredict.compute_similarities.mergeFeatureMatrix)
  * [generatePairs](#.openpredict.compute_similarities.generatePairs)
  * [balance\_data](#.openpredict.compute_similarities.balance_data)
  * [geometricMean](#.openpredict.compute_similarities.geometricMean)
  * [createFeatureDF](#.openpredict.compute_similarities.createFeatureDF)
  * [calculateCombinedSimilarity](#.openpredict.compute_similarities.calculateCombinedSimilarity)
  * [trainModel](#.openpredict.compute_similarities.trainModel)
  * [multimetric\_score](#.openpredict.compute_similarities.multimetric_score)
  * [evaluate](#.openpredict.compute_similarities.evaluate)
  * [get\_drug\_disease\_similarities](#.openpredict.compute_similarities.get_drug_disease_similarities)
* [openpredict.openpredict\_api](#.openpredict.openpredict_api)
  * [start\_api](#.openpredict.openpredict_api.start_api)
  * [get\_predict\_drug\_disease](#.openpredict.openpredict_api.get_predict_drug_disease)
* [openpredict.\_\_main\_\_](#.openpredict.__main__)
  * [main](#.openpredict.__main__.main)

<a name=".openpredict"></a>
# openpredict

<a name=".openpredict.compute_similarities"></a>
# openpredict.compute\_similarities

<a name=".openpredict.compute_similarities.adjcencydict2matrix"></a>
### adjcencydict2matrix

```python
adjcencydict2matrix(df, name1, name2)
```

Convert dict to matrix

**Arguments**:

- `df`: Dataframe
- `name1`: index name
- `name2`: columns name

<a name=".openpredict.compute_similarities.mergeFeatureMatrix"></a>
### mergeFeatureMatrix

```python
mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
```

Merge the drug and disease feature matrix

**Arguments**:

- `drugfeatfiles`: Drug features files list
- `diseasefeatfiles`: Disease features files list

<a name=".openpredict.compute_similarities.generatePairs"></a>
### generatePairs

```python
generatePairs(drug_df, disease_df, drugDiseaseKnown)
```

Generate positive and negative pairs using the Drug dataframe, the Disease dataframe and known drug-disease associations dataframe

**Arguments**:

- `drug_df`: Drug dataframe
- `disease_df`: Disease dataframe
- `drugDiseaseKnown`: Known drug-disease association dataframe

<a name=".openpredict.compute_similarities.balance_data"></a>
### balance\_data

```python
balance_data(pairs, classes, n_proportion)
```

Balance negative and positives samples

**Arguments**:

- `pairs`: Positive/negative pairs previously generated
- `classes`: Classes corresponding to the pairs
- `n_proportion`: Proportion number, e.g. 2

<a name=".openpredict.compute_similarities.geometricMean"></a>
### geometricMean

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

<a name=".openpredict.compute_similarities.createFeatureDF"></a>
### createFeatureDF

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

<a name=".openpredict.compute_similarities.calculateCombinedSimilarity"></a>
### calculateCombinedSimilarity

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

<a name=".openpredict.compute_similarities.trainModel"></a>
### trainModel

```python
trainModel(train_df, clf)
```

Train model

<a name=".openpredict.compute_similarities.multimetric_score"></a>
### multimetric\_score

```python
multimetric_score(estimator, X_test, y_test, scorers)
```

Return a dict of score for multimetric scoring

<a name=".openpredict.compute_similarities.evaluate"></a>
### evaluate

```python
evaluate(train_df, test_df, clf)
```

Evaluate

<a name=".openpredict.compute_similarities.get_drug_disease_similarities"></a>
### get\_drug\_disease\_similarities

```python
get_drug_disease_similarities()
```

The main function to run the drug-disease similarity pipeline

**Arguments**:

- `pairs_train`: Port of the OpenPredict API, defaults to 8808
- `pairs_test`: Print debug logs, defaults to False

<a name=".openpredict.openpredict_api"></a>
# openpredict.openpredict\_api

<a name=".openpredict.openpredict_api.start_api"></a>
### start\_api

```python
start_api(port=8808, debug=False)
```

Start the Translator OpenPredict API using [zalando/connexion](https://github.com/zalando/connexion) and the `openapi.yml` definition

**Arguments**:

- `port`: Port of the OpenPredict API, defaults to 8808
- `debug`: Print debug logs, defaults to False

<a name=".openpredict.openpredict_api.get_predict_drug_disease"></a>
### get\_predict\_drug\_disease

```python
get_predict_drug_disease(drug, disease)
```

Get associations predictions for drug-disease pairs

**Arguments**:

- `drug`: Drug of the predicted association
- `disease`: Disease of the predicted association

**Returns**:

Prediction result object with score

<a name=".openpredict.__main__"></a>
# openpredict.\_\_main\_\_

<a name=".openpredict.__main__.main"></a>
### main

```python
@click.group()
main(args=None)
```

Command Line Interface to run OpenPredict

