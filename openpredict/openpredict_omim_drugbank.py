
import logging
import os
import time
from datetime import datetime
import numbers
import re
import math
import random
import numpy as np
import pandas as pd
from sklearn import model_selection, tree, ensemble, svm, linear_model, neighbors, metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold
from joblib import dump, load
from rdflib import Graph, URIRef, Literal, RDF, ConjunctiveGraph, Namespace

def adjcencydict2matrix(df, name1, name2):
    """Convert dict to matrix

    :param df: Dataframe
    :param name1: index name
    :param name2: columns name
    """
    df1 = df.copy()
    df1= df1.rename(index=str, columns={name1: name2, name2: name1})
    print('📏 Dataframe size')
    print(len(df))
    df =df.append(df1)
    print(len(df))
    print(len(df))
    return df.pivot(index=name1, columns=name2)

def mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles):
    """Merge the drug and disease feature matrix

    :param drugfeatfiles: Drug features files list
    :param diseasefeatfiles: Disease features files list
    """
    print('Load and merge features files 📂')
    for i,featureFilename in enumerate(drugfeatfiles):
        print(featureFilename)
        df = pd.read_csv(featureFilename, delimiter=',')
        print (df.columns)
        cond = df.Drug1 > df.Drug2
        df.loc[cond, ['Drug1', 'Drug2']] = df.loc[cond, ['Drug2', 'Drug1']].values
        if i != 0:
            drug_df=drug_df.merge(df,on=['Drug1','Drug2'],how='inner')
            #drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            drug_df=df
    drug_df.fillna(0, inplace=True)
    
    drug_df = adjcencydict2matrix(drug_df, 'Drug1', 'Drug2')
    drug_df = drug_df.fillna(1.0)

    
    for i,featureFilename in enumerate(diseasefeatfiles):
        print(featureFilename)
        df=pd.read_csv(featureFilename, delimiter=',')
        cond = df.Disease1 > df.Disease2
        df.loc[cond, ['Disease1','Disease2']] = df.loc[cond, ['Disease2','Disease1']].values
        if i != 0:
            disease_df = disease_df.merge(df,on=['Disease1','Disease2'], how='inner')
            #drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            disease_df = df
    disease_df.fillna(0, inplace=True)
    disease_df.Disease1 = disease_df.Disease1.astype(str)
    disease_df.Disease2 = disease_df.Disease2.astype(str)
    
    disease_df = adjcencydict2matrix(disease_df, 'Disease1', 'Disease2')
    disease_df = disease_df.fillna(1.0)
    
    return drug_df, disease_df


def generatePairs(drug_df, disease_df, drugDiseaseKnown):
    """Generate positive and negative pairs using the Drug dataframe, the Disease dataframe and known drug-disease associations dataframe 

    :param drug_df: Drug dataframe
    :param disease_df: Disease dataframe
    :param drugDiseaseKnown: Known drug-disease association dataframe
    """
    drugwithfeatures = set(drug_df.columns.levels[1])
    diseaseswithfeatures = set(disease_df.columns.levels[1])
    
    drugDiseaseDict  = set([tuple(x) for x in  drugDiseaseKnown[['Drug','Disease']].values])

    commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )
    print("💊 commonDrugs: %d 🦠  commonDiseases: %d"%(len(commonDrugs),len(commonDiseases)))

    #abridged_drug_disease = [(dr,di)  for  (dr,di)  in drugDiseaseDict if dr in drugwithfeatures and di in diseaseswithfeatures ]

    #commonDrugs = set( [ dr  for dr,di in  abridged_drug_disease])
    #commonDiseases  =set([ di  for dr,di in  abridged_drug_disease])

    print("\n🥇 Gold standard, associations: %d drugs: %d diseases: %d"%(len(drugDiseaseKnown),len(drugDiseaseKnown.Drug.unique()),len(drugDiseaseKnown.Disease.unique())))
    print("\n🏷️  Drugs with features  : %d Diseases with features: %d"%(len(drugwithfeatures),len(diseaseswithfeatures)))
    print("\n♻️  commonDrugs : %d commonDiseases : %d"%(len(commonDrugs),len(commonDiseases)))

    pairs=[]
    classes=[]
    for dr in commonDrugs:
        for di in commonDiseases:
            cls = (1 if (dr,di) in drugDiseaseDict else 0)
            pairs.append((dr,di))
            classes.append(cls)
            
    return pairs, classes


def balance_data(pairs, classes, n_proportion):
    """Balance negative and positives samples

    :param pairs: Positive/negative pairs previously generated
    :param classes: Classes corresponding to the pairs
    :param n_proportion: Proportion number, e.g. 2
    """
    classes = np.array(classes)
    pairs = np.array(pairs)
    
    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[:(n_proportion*indices_true.shape[0])]
    print("\n⚖️  ➕/➖ :", len(indices_true), len(indices), len(indices_false))
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis=0) 
    
    return pairs, classes


def geometricMean(drug, disease, knownDrugDisease, drugDF, diseaseDF):
    """Compute the geometric means of a drug-disease association using previously generated dataframes

    :param drug: Drug
    :param disease: Disease
    :param knownDrugDisease: Known drug-disease associations
    :param drugDF: Drug dataframe
    :param diseaseDF: Disease dataframe
    """
    a  = drugDF.loc[knownDrugDisease[:,0]][drug].values
    b  = diseaseDF.loc[knownDrugDisease[:,1]][disease].values
    c = np.sqrt( np.multiply(a,b) )
    ix2 = (knownDrugDisease == [drug, disease])
    c[ix2[:,1]& ix2[:,0]]=0.0
    return float(max(c))


def createFeatureArray(drug, disease, knownDrugDisease, drugDFs, diseaseDFs):
    """Create the features dataframes for Spark.

    :param drug: Drug
    :param disease: Disease
    :param knownDrugDisease: Known drug-disease associations
    :param drugDFs: Drug dataframes
    :param diseaseDFs: Disease dataframes
    :return: The features dataframe 
    """
    #featureMatri x= np.empty((len(classes),totalNumFeatures), float)
    feature_array =[]
    for i,drug_col in enumerate(drugDFs.columns.levels[0]):
        for j,disease_col in enumerate(diseaseDFs.columns.levels[0]):
            drugDF = drugDFs[drug_col]
            diseaseDF = diseaseDFs[disease_col]
            feature_array.append( geometricMean( drug, disease, knownDrugDisease, drugDF, diseaseDF))
            #print (feature_series) 
    return feature_array


def sparkBuildFeatures(sc, pairs, classes, knownDrugDis,  drug_df, disease_df):
    rdd = sc.parallelize(list(zip(pairs[:,0], pairs[:,1], classes))).map(lambda x: (x[0],x[1],x[2], createFeatureArray( x[0], x[1], knownDrugDis,  drug_df, disease_df)))
    all_scores = rdd.collect()
    drug_col = drug_df.columns.levels[0]
    disease_col = disease_df.columns.levels[0]
    combined_features = ['Feature_'+dr_col+'_'+di_col for dr_col in drug_col  for di_col in disease_col]
    a = [ e[0] for e in all_scores]
    b = [ e[1] for e in all_scores]
    c = [ e[2] for e in all_scores]
    scores = [ e[3] for e in all_scores]
    df = pd.DataFrame(scores, columns=combined_features)
    df['Drug'] = a
    df['Disease' ] = b 
    df['Class' ] = c 
    return df



def createFeatureDF(pairs, classes, knownDrugDisease, drugDFs, diseaseDFs):
    """Create the features dataframes.

    :param pairs: Generated pairs
    :param classes: Classes corresponding to the pairs
    :param knownDrugDisease: Known drug-disease associations
    :param drugDFs: Drug dataframes
    :param diseaseDFs: Disease dataframes
    :return: The features dataframe 
    """
    totalNumFeatures = len(drugDFs)*len(diseaseDFs)
    #featureMatri x= np.empty((len(classes),totalNumFeatures), float)
    df =pd.DataFrame(list(zip(pairs[:,0], pairs[:,1], classes)), columns =['Drug','Disease','Class'])
    index = 0
    for i,drug_col in enumerate(drugDFs.columns.levels[0]):
        for j,disease_col in enumerate(diseaseDFs.columns.levels[0]):
            drugDF = drugDFs[drug_col]
            diseaseDF = diseaseDFs[disease_col]
            feature_series = df.apply(lambda row: geometricMean( row.Drug, row.Disease, knownDrugDisease, drugDF, diseaseDF), axis=1)
            #print (feature_series) 
            df["Feature_"+str(drug_col)+'_'+str(disease_col)] = feature_series
    return df


def calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease):
    """Compute combined similarities

    :param pairs_train: Pairs used to train
    :param pairs_test: Pairs used to test
    :param classes_train: Classes corresponding to the pairs used to train
    :param classes_test: Classes corresponding to the pairs used to test
    :param drug_df: Drug dataframe
    :param disease_df: Disease dataframe
    :param knownDrugDisease: Known drug-disease associations
    """
    try:
        # sc = pyspark.SparkContext(appName="Pi", master="spark://my-spark-spark-master:7077")
        from pyspark import SparkConf, SparkContext
        sc = SparkContext.getOrCreate()
        drug_df_bc= sc.broadcast(drug_df)
        disease_df_bc = sc.broadcast(disease_df)
        knownDrugDis_bc = sc.broadcast(knownDrugDisease)
        print ('Running Spark...')
       
        train_df= sparkBuildFeatures(sc, pairs_train, classes_train, knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)
        test_df= sparkBuildFeatures(sc, pairs_test, classes_test, knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)
        logging.info("Finishing Spark jobs...")
    except:
        logging.info("Spark cluster not found ...")
        train_df  = createFeatureDF(pairs_train, classes_train, knownDrugDisease, drug_df, disease_df)
        test_df = createFeatureDF(pairs_test, classes_test, knownDrugDisease, drug_df, disease_df)
    return train_df, test_df


def trainModel(train_df, clf):
    """Train model
    
    :param train_df: Train dataframe
    :param clf: Classifier
    """
    features = list(train_df.columns.difference(['Drug','Disease','Class']))
    X = train_df[features]
    print("Dataframe sample of training X (trainModel features):")
    print(X.head())
    y = train_df['Class']
    print(y.head())
    print('📦 Fitting classifier...')
    clf.fit(X, y)
    return clf

def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring
    
    :param estimator: Estimator
    :param X_test: X test
    :param y_test: Y test
    :param scorers: Dict of scorers
    :return: Multimetric scores
    """
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores

def evaluate(test_df, clf):
    """Evaluate the trained classifier
    :param test_df: Test dataframe
    :param clf: Classifier
    :return: Scores
    """
    features = list(test_df.columns.difference(['Drug','Disease','Class']))
    X_test =  test_df[features]
    y_test = test_df['Class']

    # https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    
    # TODO: check changes here
    # scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
    # AttributeError: module 'sklearn.metrics' has no attribute 'scorer'
    # scorers, multimetric = metrics.get_scorer._check_multimetric_scoring(clf, scoring=scoring)
    # AttributeError: 'function' object has no attribute '_check_multimetric_scoring'
    scorers = {}
    for scorer in scoring:
        scorers[scorer] = metrics.get_scorer(scorer)
    
    scores = multimetric_score(clf, X_test, y_test, scorers)
    return scores


def get_drug_disease_classifier():
    """The main function to run the drug-disease similarities pipeline, 
    and build the drug-disease classifier.
    It returns, and stores the generated classifier as a `.joblib` file 
    in the `data/models` folder,
    
    :return: Classifier of predicted similarities and scores
    """
    time_start = datetime.now()
    features_folder = "data/features/"
    resources_folder = "data/resources/"
    drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv', 
                     'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    drugfeatfiles = [ os.path.join(features_folder, fn) for fn in drugfeatfiles]
    diseasefeatfiles = [ os.path.join(features_folder, fn) for fn in diseasefeatfiles]

    # Prepare drug-disease dictionary
    drugDiseaseKnown = pd.read_csv(resources_folder + 'openpredict-omim-drug.csv',delimiter=',') 
    drugDiseaseKnown.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)
    # print(drugDiseaseKnown.head())

    # Merge feature matrix
    drug_df, disease_df = mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
    # print(drug_df.head())

    # Generate positive and negative pairs
    pairs, classes = generatePairs(drug_df, disease_df, drugDiseaseKnown)

    # Balance negative/positive samples
    n_proportion = 2
    print("\n🍱 n_proportion: " + str(n_proportion))
    pairs, classes= balance_data(pairs, classes, n_proportion)

    # Train-Test Splitting
    pairs_train, pairs_test, classes_train, classes_test = model_selection.train_test_split(pairs, classes, stratify=classes, test_size=0.2, shuffle=True)
    # print(len(pairs_train), len(pairs_test))

    # Feature extraction (Best Combined similarity)
    print('\nFeature extraction ⛏️')
    knownDrugDisease = pairs_train[classes_train==1]
    time_pairs_train = datetime.now()

    print('Store Drug and Disease DF')
    drug_df, disease_df, knownDrugDisease

    print('Pairs train runtime 🕓  ' + str(time_pairs_train - time_start))
    print('\nCalculate the combined similarity of the training pairs 🏳️‍🌈')
    train_df, test_df = calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease)
    time_calculate_similarity = datetime.now()
    print('CalculateCombinedSimilarity runtime 🕓  ' + str(time_calculate_similarity - time_pairs_train))

    # Model Training, get classifier (clf)
    print('\nModel training, getting the classifier 🏃')
    n_seed = 100
    clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, random_state=n_seed) 
    clf = trainModel(train_df, clf)
    time_training = datetime.now()
    print('Model training runtime 🕕  ' + str(time_training - time_calculate_similarity))

    # Evaluation of the trained model
    print('\nRunning evaluation of the model 📝')
    scores = evaluate(test_df, clf)
    time_evaluate = datetime.now()
    print('Evaluation runtime 🕗  ' + str(time_evaluate - time_training))

    # About 3min to run on a laptop
    print("\nTest results 🏆")
    print(scores)

    print('\nStore the model in a .joblib file 💾')
    dump(clf, 'data/models/drug_disease_model.joblib')
    # See skikit docs: https://scikit-learn.org/stable/modules/model_persistence.html

    print('Complete runtime 🕛  ' + str(datetime.now() - time_start))
    return clf, scores

def query_omim_drugbank_classifier(input_curie):
    """The main function to query the drug-disease OpenPredict classifier, 
    It queries the previously generated classifier a `.joblib` file 
    in the `data/models` folder
    
    :return: Predictions and scores
    """
    
    parsed_curie = re.search('(.*?):(.*)', input_curie)
    input_namespace = parsed_curie.group(1)
    input_id = parsed_curie.group(2)

    resources_folder = "data/resources/"
    features_folder = "data/features/"
    drugfeatfiles = ['drugs-fingerprint-sim.csv','drugs-se-sim.csv', 
                     'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv','drugs-target-seq-sim.csv']
    diseasefeatfiles =['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv' ]
    drugfeatfiles = [ os.path.join(features_folder, fn) for fn in drugfeatfiles]
    diseasefeatfiles = [ os.path.join(features_folder, fn) for fn in diseasefeatfiles]

    ## Get all DFs
    # Merge feature matrix
    drug_df, disease_df = mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles)
    drugDiseaseKnown = pd.read_csv(resources_folder + 'openpredict-omim-drug.csv',delimiter=',') 
    drugDiseaseKnown.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)

    # TODO: save json?
    drugDiseaseDict  = set([tuple(x) for x in  drugDiseaseKnown[['Drug','Disease']].values])

    drugwithfeatures = set(drug_df.columns.levels[1].tolist())
    diseaseswithfeatures = set(disease_df.columns.levels[1].tolist())

    # TODO: save json?
    commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )

    # Load classifier
    clf = load('data/models/drug_disease_model.joblib') 

    pairs=[]
    classes=[]
    if input_namespace.lower() == "drugbank":
        # Input is a drug, we only iterate on disease
        dr = input_id
        # drug_column_label = "source"
        # disease_column_label = "target"
        for di in commonDiseases:
            cls = (1 if (dr,di) in drugDiseaseDict else 0)
            pairs.append((dr,di))
            classes.append(cls)
    else: 
        # Input is a disease
        di = input_id
        # drug_column_label = "target"
        # disease_column_label = "source"
        for dr in commonDrugs:
            cls = (1 if (dr,di) in drugDiseaseDict else 0)
            pairs.append((dr,di))
            classes.append(cls)

    classes = np.array(classes)
    pairs = np.array(pairs)

    
    try:
        logging.info('Running Spark...')
        from pyspark import SparkConf, SparkContext
        sc = SparkContext.getOrCreate()
        logging.info(sc)
        drug_df_bc= sc.broadcast(drug_df)
        disease_df_bc = sc.broadcast(disease_df)
        knownDrugDis_bc = sc.broadcast(pairs[classes==1])
        test_df= sparkBuildFeatures(sc, pairs, classes, knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)

    except:
        logging.info("Spark cluster not found. Using pandas to create feature dataframes")
        test_df = createFeatureDF(pairs, classes, pairs[classes==1], drug_df, disease_df)
    

    # Get list of drug-disease pairs (should be saved somewhere from previous computer?)
    # Another API: given the type, what kind of entities exists?
    # Getting list of Drugs and Diseases:
    # commonDrugs= drugwithfeatures.intersection( drugDiseaseKnown.Drug.unique())
    # commonDiseases=  diseaseswithfeatures.intersection(drugDiseaseKnown.Disease.unique() )
    features = list(test_df.columns.difference(['Drug','Disease','Class']))
    y_proba = clf.predict_proba(test_df[features])

    prediction_df = pd.DataFrame( list(zip(pairs[:,0], pairs[:,1], y_proba[:,1])), columns =['drug','disease','score'])
    prediction_df.sort_values(by='score', inplace=True, ascending=False)
    # prediction_df = pd.DataFrame( list(zip(pairs[:,0], pairs[:,1], y_proba[:,1])), columns =[drug_column_label,disease_column_label,'score'])
    prediction_df["drug"]= "DRUGBANK:" + prediction_df["drug"]
    prediction_df["disease"] ="OMIM:" + prediction_df["disease"]


    prediction_results=prediction_df.to_json(orient='records')
    return prediction_results
