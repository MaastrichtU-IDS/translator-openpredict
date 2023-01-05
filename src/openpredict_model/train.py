import numbers
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pyspark
import typer
# from fairworkflows import FairWorkflow, is_fairstep, is_fairworkflow
# from noodles import unpack
from sklearn import linear_model, metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from openpredict import save
from openpredict.config import settings
from openpredict.rdf_utils import get_run_id
from openpredict.utils import log

cli = typer.Typer(help="Training for OpenPredict model")



# @is_fairstep(label='Prepare known drug-disease associations dictionary')
def get_known_associations():
    drugDiseaseKnown = pd.read_csv(
        os.path.join(settings.OPENPREDICT_DATA_DIR, 'resources', 'openpredict-omim-drug.csv'),
        delimiter=','
    )
    drugDiseaseKnown.rename(
        columns={'drugid': 'Drug', 'omimid': 'Disease'}, inplace=True)
    drugDiseaseKnown.Disease = drugDiseaseKnown.Disease.astype(str)
    # TODO: Translator IDs version (MONDO & CHEBI)
    # drugDiseaseKnown = pd.read_csv('openpredict/data/resources/known-drug-diseases.csv', delimiter=',')
    # print(drugDiseaseKnown.head())
    return drugDiseaseKnown


# @is_fairstep(label='Prepare or load drug-disease features dataframes')
def get_drug_disease_features(from_model_id: str):
    if from_model_id == 'openpredict_baseline':
        print('🏗 Build the model from scratch')
        # Start from scratch (merge feature matrixes)
        baseline_features_folder = "baseline_features/"
        drugfeatfiles = ['drugs-fingerprint-sim.csv', 'drugs-se-sim.csv',
                         'drugs-ppi-sim.csv', 'drugs-target-go-sim.csv', 'drugs-target-seq-sim.csv']
        diseasefeatfiles = ['diseases-hpo-sim.csv',  'diseases-pheno-sim.csv']
        drugfeatfiles = [os.path.join(settings.OPENPREDICT_DATA_DIR,
            baseline_features_folder, fn) for fn in drugfeatfiles]
        diseasefeatfiles = [os.path.join(settings.OPENPREDICT_DATA_DIR,
            baseline_features_folder, fn) for fn in diseasefeatfiles]
        # baseline_features_folder = "data/baseline_features/"
        # TODO: Translator IDs version (MONDO & CHEBI)
        drug_df, disease_df = mergeFeatureMatrix(
            drugfeatfiles, diseasefeatfiles)
    else:
        print(type(from_model_id))
        print(f"📥 Loading the features tensor from {settings.OPENPREDICT_DATA_DIR}/features/{str(from_model_id)}_features.pickle")

        (drug_df, disease_df) = pickle.load(open(
            f"{settings.OPENPREDICT_DATA_DIR}/features/{str(from_model_id)}_features.pickle",
            "rb"
        ))
    print("Drug Features ", drug_df.columns.levels[0])
    print("Disease Features ", disease_df.columns.levels[0])
    return (drug_df, disease_df)


# @is_fairstep(label='Train test splitting')
def train_test_splitting(n_fold, pairs, classes, drug_df, disease_df):
    time_start = datetime.now()
    # Train-Test Splitting
    n_seed = 101
    if n_fold == 1:
        rs = StratifiedShuffleSplit(
            n_splits=1, test_size=.2, random_state=n_seed)
        cv = rs.split(pairs, classes)
    else:
        skf = StratifiedKFold(
            n_splits=n_fold, shuffle=True, random_state=n_seed)
        cv = skf.split(pairs, classes)
    cv_results = pd.DataFrame()

    for i, (train, test) in enumerate(cv):
        # pairs_train, pairs_test, classes_train, classes_test = model_selection.train_test_split(
        #    pairs, classes, stratify=classes, test_size=0.2, shuffle=True)
        # print(len(pairs_train), len(pairs_test))

        # Feature extraction (Best Combined similarity)
        print('\nFeature extraction ⛏️')
        print('Fold', i+1)
        pairs_train = pairs[train]
        classes_train = classes[train]
        pairs_test = pairs[test]
        classes_test = classes[test]

        knownDrugDisease = pairs_train[classes_train == 1]
        time_pairs_train = datetime.now()

        print('Pairs train runtime 🕓  ' + str(time_pairs_train - time_start))
        print('\nCalculate the combined similarity of the training pairs 🏳️‍🌈')
        train_df, test_df = calculateCombinedSimilarity(
            pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease)
        time_calculate_similarity = datetime.now()
        print('CalculateCombinedSimilarity runtime 🕓  ' +
              str(time_calculate_similarity - time_pairs_train))

        # Model Training, get classifier (clf)
        print('\nModel training, getting the classifier 🏃')
        clf = linear_model.LogisticRegression(penalty=hyper_params['penalty'],
                                              dual=hyper_params['dual'], tol=hyper_params['tol'],
                                              C=hyper_params['C'], random_state=hyper_params['random_state'])
        clf, sample_data = train_classifier(train_df, clf)
        time_training = datetime.now()
        print('Model training runtime 🕕  ' +
              str(time_training - time_calculate_similarity))

        # Evaluation of the trained model
        print('\nRunning evaluation of the model 📝')
        scores = evaluate(test_df, clf)
        time_evaluate = datetime.now()
        print('Evaluation runtime 🕗  ' + str(time_evaluate - time_training))

        # About 3min to run on a laptop
        print("\nTest results 🏆")
        print(scores)
        cv_results = cv_results.append(scores, ignore_index=True)

    scores = cv_results.mean()
    print("\n " + str(n_fold) + "-fold CV - Avg Test results 🏆")
    print(scores)
    return scores


# @is_fairstep(label='Get the classifier')
def get_classifier(hyper_params):
    return linear_model.LogisticRegression(
        penalty=hyper_params['penalty'],
        dual=hyper_params['dual'],
        tol=hyper_params['tol'],
        C=hyper_params['C'],
        random_state=hyper_params['random_state']
    )


hyper_params = {
    'penalty': 'l2',
    'dual': False,
    'tol': 0.0001,
    'C': 1.0,
    'random_state': 100
}
sc = None

# @is_fairstep(label='Get Spark context', is_script_task=True)
def get_spark_context():
    """Get Spark context, either from Spark Master URL to a Spark cluster
    If not URL is provided, then it will try to run Spark locally

    :return: Spark context
    """
    spark_master_url = os.getenv('SPARK_MASTER_URL')
    sc = None

    if os.getenv('SPARK_HOME'):
        # Do not try to run Spark if SPARK_HOME env variable not set
        # import findspark
        # findspark.init(os.getenv('SPARK_HOME'))
        # sc = pyspark.SparkContext(appName="Pi", master='local[*]')

        if spark_master_url and sc is None:
            log.info(
                'SPARK_MASTER_URL provided, connecting to the Spark cluster ✨')
            # e.g. spark://my-spark-spark-master:7077
            sc = pyspark.SparkContext(appName="Pi", master=spark_master_url)
            log.info(sc)
        else:
            # Most of the time use local Spark available in docker container
            try:
                log.info(
                    'SPARK_MASTER_URL not provided, trying to start Spark locally ✨')
                sc = pyspark.SparkContext.getOrCreate()
                # sc = pyspark.SparkContext(appName="Pi", master='local[*]')
                log.info(sc)
            except Exception as e:
                log.warning(e)
                log.info(
                    "⚠️ Could not start a Spark cluster locally. Using pandas to handle dataframes 🐼")

    else:
        log.info(
            'SPARK_HOME environment variable not found, using pandas to handle dataframes 🐼')
    return sc
    # Old way:
    #     import findspark
    #     from pyspark import SparkConf, SparkContext
    #     findspark.init()

    #     config = SparkConf()
    #     config.setMaster("local[*]")
    #     config.set("spark.executor.memory", "5g")
    #     config.set('spark.driver.memory', '5g')
    #     config.set("spark.memory.offHeap.enabled",True)
    #     config.set("spark.memory.offHeap.size","5g")
    #     sc = SparkContext(conf=config, appName="OpenPredict")


# @is_fairstep(label='Convert a dictionary to a matrix', is_script_task=True)
def adjcencydict2matrix(df, name1, name2):
    """Convert dict to matrix

    :param df: Dataframe
    :param name1: index name
    :param name2: columns name
    """
    df1 = df.copy()
    df1 = df1.rename(index=str, columns={name1: name2, name2: name1})
    print('📏 Dataframe size')
    print(len(df))
    df = df.append(df1)
    print(len(df))
    print(len(df))
    return df.pivot(index=name1, columns=name2)


# @is_fairstep(label='Merge the drug and disease feature matrix', is_script_task=True)
def mergeFeatureMatrix(drugfeatfiles, diseasefeatfiles):
    """Merge the drug and disease feature matrix

    :param drugfeatfiles: Drug features files list
    :param diseasefeatfiles: Disease features files list
    """
    print('Load and merge features files 📂')
    drug_df = None
    for i, featureFilename in enumerate(drugfeatfiles):
        print(featureFilename)
        df = pd.read_csv(featureFilename, delimiter=',')
        print(df.columns)
        cond = df.Drug1 > df.Drug2
        df.loc[cond, ['Drug1', 'Drug2']
               ] = df.loc[cond, ['Drug2', 'Drug1']].values
        if i != 0:
            drug_df = drug_df.merge(df, on=['Drug1', 'Drug2'], how='inner')
            # drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            drug_df = df
    drug_df.fillna(0, inplace=True)

    drug_df = adjcencydict2matrix(drug_df, 'Drug1', 'Drug2')
    drug_df = drug_df.fillna(1.0)

    disease_df = None
    for i, featureFilename in enumerate(diseasefeatfiles):
        print(featureFilename)
        df = pd.read_csv(featureFilename, delimiter=',')
        cond = df.Disease1 > df.Disease2
        df.loc[cond, ['Disease1', 'Disease2']
               ] = df.loc[cond, ['Disease2', 'Disease1']].values
        if i != 0:
            disease_df = disease_df.merge(
                df, on=['Disease1', 'Disease2'], how='outer')
            # drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            disease_df = df
    disease_df.fillna(0, inplace=True)
    disease_df.Disease1 = disease_df.Disease1.astype(str)
    disease_df.Disease2 = disease_df.Disease2.astype(str)

    disease_df = adjcencydict2matrix(disease_df, 'Disease1', 'Disease2')
    disease_df = disease_df.fillna(1.0)

    return drug_df, disease_df


# @is_fairstep(label='Generate positive and negative pairs using the Drug dataframe, the Disease dataframe and known drug-disease associations dataframe', is_script_task=True)
def generatePairs(drug_df, disease_df, drugDiseaseKnown):
    """Generate positive and negative pairs using the Drug dataframe,
    the Disease dataframe and known drug-disease associations dataframe

    :param drug_df: Drug dataframe
    :param disease_df: Disease dataframe
    :param drugDiseaseKnown: Known drug-disease association dataframe
    """
    drugwithfeatures = set(drug_df.columns.levels[1])
    diseaseswithfeatures = set(disease_df.columns.levels[1])

    drugDiseaseDict = {
        tuple(x) for x in drugDiseaseKnown[['Drug', 'Disease']].values}

    commonDrugs = drugwithfeatures.intersection(drugDiseaseKnown.Drug.unique())
    commonDiseases = diseaseswithfeatures.intersection(
        drugDiseaseKnown.Disease.unique())
    print("💊 commonDrugs: %d 🦠  commonDiseases: %d" %
          (len(commonDrugs), len(commonDiseases)))

    # abridged_drug_disease = [(dr,di)  for  (dr,di)  in drugDiseaseDict if dr in drugwithfeatures and di in diseaseswithfeatures ]

    # commonDrugs = set( [ dr  for dr,di in  abridged_drug_disease])
    # commonDiseases  =set([ di  for dr,di in  abridged_drug_disease])

    print("\n🥇 Gold standard, associations: %d drugs: %d diseases: %d" % (len(drugDiseaseKnown), len(
        drugDiseaseKnown.Drug.unique()), len(drugDiseaseKnown.Disease.unique())))
    print("\n🏷️  Drugs with features  : %d Diseases with features: %d" %
          (len(drugwithfeatures), len(diseaseswithfeatures)))
    print("\n♻️  commonDrugs : %d commonDiseases : %d" %
          (len(commonDrugs), len(commonDiseases)))

    pairs = []
    classes = []
    for dr in commonDrugs:
        for di in commonDiseases:
            cls = (1 if (dr, di) in drugDiseaseDict else 0)
            pairs.append((dr, di))
            classes.append(cls)

    return pairs, classes


# @is_fairstep(label='Balance negative and positives samples', is_script_task=True)
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
    print("\n⚖️  +/- :", len(indices_true), len(indices), len(indices_false))
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis=0)

    return pairs, classes


# @is_fairstep(label='Compute the geometric means of a drug-disease association using previously generated dataframes', is_script_task=True)
def geometricMean(drug, disease, knownDrugDisease, drugDF, diseaseDF):
    """Compute the geometric means of a drug-disease association using previously generated dataframes

    :param drug: Drug
    :param disease: Disease
    :param knownDrugDisease: Known drug-disease associations
    :param drugDF: Drug dataframe
    :param diseaseDF: Disease dataframe
    """
    a = drugDF.loc[knownDrugDisease[:, 0]][drug].values
    b = diseaseDF.loc[knownDrugDisease[:, 1]][disease].values
    c = np.sqrt(np.multiply(a, b))
    ix2 = (knownDrugDisease == [drug, disease])
    c[ix2[:, 1] & ix2[:, 0]] = 0.0
    if len(c) == 0:
        return 0.0
    return float(max(c))


# @is_fairstep(label='Create the features dataframes for Spark', is_script_task=True)
def createFeatureArray(drug, disease, knownDrugDisease, drugDFs, diseaseDFs):
    """Create the features dataframes for Spark.

    :param drug: Drug
    :param disease: Disease
    :param knownDrugDisease: Known drug-disease associations
    :param drugDFs: Drug dataframes
    :param diseaseDFs: Disease dataframes
    :return: The features dataframe
    """
    # featureMatri x= np.empty((len(classes),totalNumFeatures), float)
    feature_array = []
    for _i, drug_col in enumerate(drugDFs.columns.levels[0]):
        for _j, disease_col in enumerate(diseaseDFs.columns.levels[0]):
            drugDF = drugDFs[drug_col]
            diseaseDF = diseaseDFs[disease_col]
            feature_array.append(geometricMean(
                drug, disease, knownDrugDisease, drugDF, diseaseDF))
            # print (feature_series)
    return feature_array


# @is_fairstep(label='Create the feature matrix for Spark', is_script_task=True)
def sparkBuildFeatures(sc, pairs, classes, knownDrugDis,  drug_df, disease_df):
    """Create the feature matrix for Spark.

    :param sc: Spark context
    :param pairs: Generated pairs
    :param classes: Classes corresponding to the pairs
    :param knownDrugDisease: Known drug-disease associations
    :param drugDFs: Drug dataframes
    :param diseaseDFs: Disease dataframes
    :return: The features dataframe
    """

    rdd = sc.parallelize(list(zip(pairs[:, 0], pairs[:, 1], classes))).map(lambda x: (
        x[0], x[1], x[2], createFeatureArray(x[0], x[1], knownDrugDis,  drug_df, disease_df)))
    all_scores = rdd.collect()
    drug_col = drug_df.columns.levels[0]
    disease_col = disease_df.columns.levels[0]
    combined_features = ['Feature_'+dr_col+'_' +
                         di_col for dr_col in drug_col for di_col in disease_col]
    a = [e[0] for e in all_scores]
    b = [e[1] for e in all_scores]
    c = [e[2] for e in all_scores]
    scores = [e[3] for e in all_scores]
    df = pd.DataFrame(scores, columns=combined_features)
    df['Drug'] = a
    df['Disease'] = b
    df['Class'] = c
    return df


# @is_fairstep(label='Create the features dataframes', is_script_task=True)
def createFeatureDF(pairs, classes, knownDrugDisease, drugDFs, diseaseDFs):
    """Create the features dataframes.

    :param pairs: Generated pairs
    :param classes: Classes corresponding to the pairs
    :param knownDrugDisease: Known drug-disease associations
    :param drugDFs: Drug dataframes
    :param diseaseDFs: Disease dataframes
    :return: The features dataframe
    """
    len(drugDFs)*len(diseaseDFs)
    # featureMatri x= np.empty((len(classes),totalNumFeatures), float)
    df = pd.DataFrame(list(zip(pairs[:, 0], pairs[:, 1], classes)), columns=[
                      'Drug', 'Disease', 'Class'])
    for _i, drug_col in enumerate(drugDFs.columns.levels[0]):
        for _j, disease_col in enumerate(diseaseDFs.columns.levels[0]):
            drugDF = drugDFs[drug_col]
            diseaseDF = diseaseDFs[disease_col]
            feature_series = df.apply(lambda row: geometricMean(
                row.Drug, row.Disease, knownDrugDisease, drugDF, diseaseDF), axis=1)
            # print (feature_series)
            df["Feature_"+str(drug_col)+'_'+str(disease_col)] = feature_series
    return df


# @is_fairstep(label='Compute combined similarities', is_script_task=True)
def calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease):
    """Compute combined similarities. Use Spark if available for speed, otherwise use pandas

    :param pairs_train: Pairs used to train
    :param pairs_test: Pairs used to test
    :param classes_train: Classes corresponding to the pairs used to train
    :param classes_test: Classes corresponding to the pairs used to test
    :param drug_df: Drug dataframe
    :param disease_df: Disease dataframe
    :param knownDrugDisease: Known drug-disease associations
    """
    spark_context = get_spark_context()
    if spark_context:
        drug_df_bc = spark_context.broadcast(drug_df)
        disease_df_bc = spark_context.broadcast(disease_df)
        knownDrugDis_bc = spark_context.broadcast(knownDrugDisease)
        log.info('Running Spark ✨')
        train_df = sparkBuildFeatures(spark_context, pairs_train, classes_train,
                                      knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)
        test_df = sparkBuildFeatures(spark_context, pairs_test, classes_test,
                                     knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)
        log.info("Finishing Spark jobs 🏁")
    else:
        log.info("Spark cluster not found, using pandas 🐼")
        train_df = createFeatureDF(
            pairs_train, classes_train, knownDrugDisease, drug_df, disease_df)
        test_df = createFeatureDF(
            pairs_test, classes_test, knownDrugDisease, drug_df, disease_df)

    return train_df, test_df


# @is_fairstep(label='Train classifier', is_script_task=True)
def train_classifier(train_df, clf):
    """Train classifier

    :param train_df: Train dataframe
    :param clf: Classifier
    """
    features = list(train_df.columns.difference(['Drug', 'Disease', 'Class']))
    X = train_df[features]
    print("Dataframe sample of training X (train_classifier features):")
    print(X.head())
    y = train_df['Class']
    print(y.head())
    print('📦 Fitting classifier...')
    clf.fit(X, y)
    return clf, X


# @is_fairstep(label='Return a dict of score for multimetric scoring', is_script_task=True)
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


# @is_fairstep(label='Evaluate the trained classifier', is_script_task=True)
def evaluate(test_df, clf):
    """Evaluate the trained classifier
    :param test_df: Test dataframe
    :param clf: Classifier
    :return: Scores
    """
    features = list(test_df.columns.difference(['Drug', 'Disease', 'Class']))
    X_test = test_df[features]
    y_test = test_df['Class']

    # https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation
    scoring = ['precision', 'recall', 'accuracy',
               'roc_auc', 'f1', 'average_precision']

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


# @is_fairstep(label='Create features dataframes', is_script_task=True)
def createFeaturesSparkOrDF(pairs, classes, drug_df, disease_df):
    """Create features dataframes. Use Spark if available for speed, otherwise use pandas
    :param pairs: pairs
    :param classes: classes
    :param drug_df: drug
    :param disease_df: disease dataframe
    :return: Feature dataframe
    """
    spark_context = get_spark_context()
    if spark_context:
        log.info('Running Spark ✨')
        drug_df_bc = spark_context.broadcast(drug_df)
        disease_df_bc = spark_context.broadcast(disease_df)
        knownDrugDis_bc = spark_context.broadcast(pairs[classes == 1])
        feature_df = sparkBuildFeatures(
            spark_context, pairs, classes, knownDrugDis_bc.value,  drug_df_bc.value, disease_df_bc.value)
        log.info("Finishing Spark jobs 🏁")
    else:
        log.info("Spark cluster not found, using pandas 🐼")
        feature_df = createFeatureDF(
            pairs, classes, pairs[classes == 1], drug_df, disease_df)
    return feature_df




@cli.command(help='Train an existing OpenPredict model')
def train(
    from_model_id: str = 'openpredict_baseline',
    embedding_file: str = None,
    emb_name: str = None,
    types: str = 'Drugs',
    create_model_id: str = None,
):
    if not embedding_file:
        train_model(from_model_id)
    else:
        add_embedding(embedding_file, emb_name, types, from_model_id)

    # if not embedding_file:
    #     workflow = FairWorkflow.from_function(train_model)
    #     result, prov = workflow.execute(from_model_id)
    # else:
    #     workflow = FairWorkflow.from_function(add_embedding)
    #     result, prov = workflow.execute(embedding_file, emb_name, types, from_model_id)

    # workflow.publish_as_nanopub(use_test_server=True, publish_steps=True)
    # prov.publish_as_nanopub(use_test_server=True)

    # workflow._rdf.serialize(f"models/{from_model_id}.workflow.trig", format="trig")
    # prov._rdf.serialize(f"models/{from_model_id}.prov.trig", format="trig")



# @is_fairworkflow(label='OpenPredict model training workflow')
def train_model(from_model_id: str = 'openpredict_baseline'):
    """The main function to run the drug-disease similarities pipeline,
    and train the drug-disease classifier.
    It returns, and stores the generated classifier as a pickle file
    in the `data/models` folder,

    :param from_scratch: Train the model for scratch (True by default)
    :return: Classifier of predicted similarities and scores
    """
    # time_start = datetime.now()
    n_fold = 5
    # n_fold = 1
    print(type(from_model_id))
    # if isinstance(from_model_id, inspect._empty):
    # if type(from_model_id) == "<class 'inspect._empty'>":
    #     # print("Empty model provided by fairworkflow, skipping. cf. https://github.com/fair-workflows/fairworkflows/blob/main/fairworkflows/fairworkflow.py#L503")
    #     return {}

    drugDiseaseKnown = get_known_associations()
    # return drugDiseaseKnown
    # (drug_df, disease_df) = get_drug_disease_features(from_model_id)
    (drug_df, disease_df) = get_drug_disease_features(from_model_id)

    # Generate positive and negative pairs
    pairs, classes = generatePairs(drug_df, disease_df, drugDiseaseKnown)

    # Balance negative/positive samples
    n_proportion = 2
    # print("\n🍱 n_proportion: " + str(n_proportion))
    pairs, classes = balance_data(pairs, classes, n_proportion)

    scores = train_test_splitting(n_fold, pairs, classes, drug_df, disease_df)

    # print("\n Train the final model using all dataset")
    # final_training = datetime.now()
    train_df = createFeaturesSparkOrDF(pairs, classes, drug_df, disease_df)

    clf = get_classifier(hyper_params)

    # penalty: HyperParameter , l2: HyperParameterSetting
    # Implementation: LogisticRegression
    clf, sample_data = train_classifier(train_df, clf)
    # print(f"Final model training runtime 🕕 {str(datetime.now() - final_training)}")

    # if from_model_id == 'openpredict_baseline':
    #     pickle.dump(
    #         (drug_df, disease_df),
    #         open(
    #             f"{settings.OPENPREDICT_DATA_DIR}/features/{from_model_id}_features.pickle",
    #             "wb"
    #         )
    #     )
    #     print('New embedding based similarity was added to the similarity tensor and dataframes with new features are store in data/features/openpredict_baseline_features.pickle')
    #     # See skikit docs: https://scikit-learn.org/stable/modules/model_persistence.html

    # print('Complete runtime 🕛  ' + str(datetime.now() - time_start))

    loaded_model = save(
        model=clf,
        path="models/openpredict_baseline",
        sample_data=sample_data,
        hyper_params=hyper_params,
        scores=scores,
    )
    return loaded_model




# @is_fairworkflow(label='Add new embeddings to the OpenPredict model, and train the new model')
def add_embedding(
    embedding_file: str,
    emb_name: str,
    types: str = 'Drugs',
    # description: str,
    from_model_id: str = "openpredict_baseline"
):
    """Add embedding to the drug similarity matrix dataframe

    :param embedding_file: JSON file containing records ('entity': id, 'embdding': array of numbers )
    :param emb_name: new column name to be added
    :param types: types in the embedding vector ['Drugs', 'Diseases', 'Both']
    :param description: description of the embedding provenance
    """
    emb_df = pd.read_json(embedding_file, orient='records')

    # print(emb_df.head())
    # emb_df = pd.read_csv(embedding_file)
    emb_df.entity = emb_df.entity.str.replace('DRUGBANK:', '')
    emb_df.entity = emb_df.entity.str.replace('OMIM:', '')
    # print (emb_df.head())
    emb_df.set_index('entity', inplace=True)

    print(emb_df.head())
    emb_size = len(emb_df.iloc[0]['embedding'])
    print('Embedding dimension', emb_size)

    # TODO: now also save the feature dataframe for each run to be able to add embedding to any run?
    # Or can we just use the models/run_id.pickle file instead of having 2 files for 1 run?
    print('📥 Loading features file: ' +
          f"{settings.OPENPREDICT_DATA_DIR}/features/{from_model_id}_features.pickle")
    (drug_df, disease_df) = pickle.load(open(
        f"{settings.OPENPREDICT_DATA_DIR}/features/{from_model_id}_features.pickle",
        "rb"
    ))

    if types == 'Drugs':
        names = drug_df.columns.levels[1]
        ["Drug1", "Drug2", emb_name]
    else:
        names = disease_df.columns.levels[1]
        ["Disease1", "Disease2", emb_name]

    entity_exist = [d for d in names if d in emb_df.index]
    print("Number of drugs that do not exist in the embedding ",
          len(names) - len(entity_exist))
    # Copy only drug entity embeddings
    embedding_df = emb_df.copy()
    emb_matrix = np.empty(shape=(0, emb_size))
    for d in names:
        # add zeros values for drugs that do not exist in the embedding
        if d not in emb_df.index:
            emb_matrix = np.vstack([emb_matrix, np.zeros(emb_size)])
        else:
            if len(embedding_df.loc[d]['embedding']) != emb_size:
                print(embedding_df.loc[d]['embedding'])
                embedding_df.loc[d]['embedding'] = embedding_df.loc[d]['embedding'][0]
            emb_matrix = np.vstack(
                [emb_matrix, embedding_df.loc[d]['embedding']])
    # calculate cosine similarity for given embedding
    sim_mat = cosine_similarity(emb_matrix, emb_matrix)
    # convert to DF
    df_sim = pd.DataFrame(sim_mat, index=names, columns=names)
    # if there is NA (It is the case when both pairs have zero values-no embedding exist)
    df_sim = df_sim.fillna(0.0)
    # make multi-index dataframe adding a new column with given embedding name
    df_sim_m = pd.concat([df_sim], axis=1, keys=[emb_name])
    # finally concatenate the embedding-based similarity to other drug similarity matrix
    print(df_sim_m.sample(5))

    # add to the similarity tensor

    if types == "Drugs":
        df_sim_m.index = drug_df.index
        drug_df = pd.concat([drug_df, df_sim_m],  axis=1)
    elif types == "Diseases":
        df_sim_m.index = disease_df.index
        disease_df = pd.concat([disease_df, df_sim_m],  axis=1)

    run_id = get_run_id()
    pickle.dump(
        (drug_df, disease_df),
        open(
            f"{settings.OPENPREDICT_DATA_DIR}/features/{run_id}_features.pickle",
            "wb"
        )
    )

    # added_feature_uri = add_feature_metadata(emb_name, description, types)
    # train the model
    loaded_model = train_model(run_id)

    return run_id, loaded_model




if __name__ == '__main__':
    cli()
    # When directly executed as script
    # print(train_model())

    # loaded = load('models/openpredict_baseline')
    # import pickle
    # model = pickle.load(open("models/openpredict_baseline", "rb"))
    # features = pickle.load(open("models/openpredict_baseline.features", "rb"))
    # sample_data = pickle.load(open("../models_BACK/sample_data.pickle", "rb"))
    # scores = pickle.load(open("models_BACK/scores.pickle", "rb"))
    # print("model")
    # print(model)
    # print("sample_data")
    # print(sample_data)
    # print("features")
    # print(features)
    # save(
    #     model=model,
    #     path=f"models/openpredict_baseline_mlem",
    #     sample_data=sample_data,
    #     features=features,
    #     hyper_params=hyper_params,
    #     scores=scores,
    # )
