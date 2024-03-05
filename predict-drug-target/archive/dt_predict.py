import numbers
import random
from datetime import date

import findspark
import numpy
import numpy as np
import pandas as pd
import torch
from pyspark import SparkConf, SparkContext
from sklearn import ensemble, metrics
from sklearn.model_selection import StratifiedKFold

# !pip install git+https://github.com/facebookresearch/esm.git


def loadProteinEmbeddings(path, embedding_layer=33, use_mean=True):
    import glob

    files = sorted(glob.glob(path + "/*.pt"))

    protein_embeddings = []
    protein_labels = []
    for file in files:
        try:
            label = file.split(".pt")[0].split("/")[-1]
            embs = torch.load(file)
            representation = "representations"
            if use_mean:
                representation = "mean_representations"
            emb = embs[representation][embedding_layer]  # this is a torch.Tensor
            # the mean_representations are the same size for each entry
            # the per_tok embedding, it is seq size by 1280
            protein_embeddings.append(emb)
            protein_labels.append(label)
        except Exception:
            print(f"unable to open {file}")
            continue

    # vectors = np.stack([emb.mean(axis=0) for emb in embeddings])
    Xs = torch.stack(protein_embeddings, dim=0).numpy()  # numpy.ndarray 3775 x 1280

    target_labels_df = pd.DataFrame(protein_labels, columns=["target"])
    target_embeddings_df = pd.DataFrame(Xs)
    target_embeddings_df["target"] = target_labels_df["target"]
    # target_df = target_labels_df.merge(target_embeddings_df, how='cross')
    return target_embeddings_df


def loadDrugEmbeddings(drug_labels_file, drug_embeddings_file):
    drug_labels_df = pd.read_csv(drug_labels_file)
    drug_labels_df = drug_labels_df.drop(columns=["smiles", "drugs"])

    o = np.load(drug_embeddings_file)  # numpy.lib.npyio.NpzFile
    files = o.files  # 5975 files
    embeddings = []
    for file in files:
        # print(file)
        emb = o[file]  # emb 'numpy.ndarray' n length x 512
        embeddings.append(emb)
        # print(emb.shape)
    vectors = np.stack([emb.mean(axis=0) for emb in embeddings])
    # vectors.shape # 5975, 512
    df = pd.DataFrame(vectors)
    drug_df = drug_labels_df.merge(df, left_index=True, right_index=True)
    return drug_df


def loadDrugTargets(path):
    df = pd.read_csv(path)
    return df


def generateDTPairs(dt_df):
    dtKnown = set([tuple(x) for x in dt_df[["drug", "target"]].values])
    pairs = list()
    labels = list()

    drugs = set(dt_df.drug.unique())
    targets = set(dt_df.target.unique())
    for d in drugs:
        for t in targets:
            if (d, t) in dtKnown:
                label = 1
            else:
                label = 0

            pairs.append((d, t))
            labels.append(label)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError(
                "scoring must return a number, got %s (%s) " "instead. (scorer=%s)" % (str(score), type(score), name)
            )
    return scores


def balance_data(pairs, classes, n_proportion):
    classes = np.array(classes)
    pairs = np.array(pairs)

    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[: (n_proportion * indices_true.shape[0])]

    print(f"True positives: {len(indices_true)}")
    print(f"True negatives: {len(indices_false)}")
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis=0)

    return pairs, classes


def get_scores(clf, X_new, y_new):
    scoring = ["precision", "recall", "accuracy", "roc_auc", "f1", "average_precision"]
    scorers = metrics._scorer._check_multimetric_scoring(clf, scoring=scoring)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores


def crossvalid(train_df, test_df, clfs, run_index, fold_index):
    features_cols = train_df.columns.difference(["drug", "target", "Class"])
    X = train_df[features_cols].values
    y = train_df["Class"].values.ravel()

    X_new = test_df[features_cols].values
    y_new = test_df["Class"].values.ravel()

    results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(X, y)
        row = {}
        row["run"] = run_index
        row["fold"] = fold_index
        row["method"] = name
        scores = get_scores(clf, X_new, y_new)
        row.update(scores)

        df = pd.DataFrame.from_dict([row])
        results = pd.concat([results, df], ignore_index=True)

    return results  # , sclf_scores


def cv_run(run_index, pairs, classes, embedding_df, train, test, fold_index, clfs):
    # print( f"Run: {run_index} Fold: {fold_index} Train size: {len(train)} Test size: {len(test)}")
    train_df = pd.DataFrame(
        list(zip(pairs[train, 0], pairs[train, 1], classes[train])), columns=["drug", "target", "Class"]
    )
    test_df = pd.DataFrame(
        list(zip(pairs[test, 0], pairs[test, 1], classes[test])), columns=["drug", "target", "Class"]
    )

    train_df = train_df.merge(embedding_df["drug"], left_on="drug", right_on="drug").merge(
        embedding_df["target"], left_on="target", right_on="target"
    )
    test_df = test_df.merge(embedding_df["drug"], left_on="drug", right_on="drug").merge(
        embedding_df["target"], left_on="target", right_on="target"
    )

    all_scores = crossvalid(train_df, test_df, clfs, run_index, fold_index)
    if run_index == 1 and fold_index == 0:
        print(", ".join(all_scores.columns))
    print(all_scores.to_string(header=False, index=False))

    return all_scores


def cvSpark(sc, run_index, pairs, classes, cv, embedding_df, clfs):
    if sc:
        rdd = sc.parallelize(cv).map(lambda x: cv_run(run_index, pairs, classes, embedding_df, x[0], x[1], x[2], clfs))
        all_scores = rdd.collect()
    else:
        all_scores = pd.DataFrame()
        for x in cv:
            scores = cv_run(run_index, pairs, classes, embedding_df, x[0], x[1], x[2], clfs)
            all_scores = pd.concat([all_scores, scores], ignore_index=True)

    return all_scores


def kfoldCV(sc, pairs_all, classes_all, embedding_df, clfs, n_run, n_fold, n_proportion, n_seed):
    if sc:
        bc_embedding_df = sc.broadcast(embedding_df)

    scores_df = pd.DataFrame()
    for r in range(1, n_run + 1):
        n_seed += r
        random.seed(n_seed)
        np.random.seed(n_seed)
        pairs, classes = balance_data(pairs_all, classes_all, n_proportion)

        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=n_seed)
        cv = skf.split(pairs, classes)

        pairs_classes = (pairs, classes)
        cv_list = [(train, test, k) for k, (train, test) in enumerate(cv)]

        if sc:
            bc_pairs_classes = sc.broadcast(pairs_classes)
            scores = cvSpark(
                sc, r, bc_pairs_classes.value[0], bc_pairs_classes.value[1], cv_list, bc_embedding_df.value, clfs
            )
            for score in scores:
                scores_df = pd.concat([scores_df, score], ignore_index=True)
        else:
            scores = cvSpark(sc, r, pairs_classes[0], pairs_classes[1], cv_list, embedding_df, clfs)
            scores_df = pd.concat([scores_df, scores], ignore_index=True)
    return scores_df


######

embeddings = {}
protein_embeddings_path = "./data/vectors/drugbank_targets_esm2_l33_mean"
embeddings["target"] = loadProteinEmbeddings(protein_embeddings_path)

drug_labels_path = "./data/download/drugbank_drugs.csv"
drug_embeddings_path = "./data/vectors/drugbank_smiles.npz"
drug_embeddings = loadDrugEmbeddings(drug_labels_path, drug_embeddings_path)
embeddings["drug"] = drug_embeddings

drug_target_path = "./data/download/drugbank_drug_targets.csv"
dt_df = loadDrugTargets(drug_target_path)

today = date.today()
results_file = f"./data/results/drugbank_drug_targets_scores_{today}.csv"
agg_results_file = f"./data/results/drugbank_drug_targets_agg_{today}.csv"

pairs, labels = generateDTPairs(dt_df)
ndrugs = len(embeddings["drug"])
ntargets = len(embeddings["target"])
print(f"Drugs: {ndrugs}")
print(f"Targets: {ntargets}")
unique, counts = numpy.unique(labels, return_counts=True)
ndrugtargets = counts[1]
print(f"Drug-Targets: {ndrugtargets}")

# nb_model = GaussianNB()
# lr_model = linear_model.LogisticRegression()
rf_model = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)

# clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model)]
clfs = [("Random Forest", rf_model)]

# Spark
sc = False
if sc:
    findspark.init()
    # sc.stop()
    config = SparkConf()
    config.setMaster("local")
    config.set("spark.executor.memory", "15g")
    config.set("spark.driver.memory", "20g")
    config.set("spark.memory.offHeap.enabled", True)
    config.set("spark.memory.offHeap.size", "15g")
    sc = SparkContext(conf=config)
    print(sc)

n_seed = 100
n_fold = 10
n_run = 2
n_proportion = 1
all_scores_df = kfoldCV(sc, pairs, labels, embeddings, clfs, n_run, n_fold, n_proportion, n_seed)
all_scores_df.to_csv(results_file, sep=",", index=False)

agg_df = all_scores_df.groupby(["method", "run"]).mean().groupby("method").mean()
agg_df.to_csv(agg_results_file, sep=",", index=False)
print("overall:")
print(agg_df)

if sc:
    sc.stop()
