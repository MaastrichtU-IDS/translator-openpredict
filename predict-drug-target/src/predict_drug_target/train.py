"""Common functions for training the models"""
import gc
import numbers
import os
import pickle
import time
import random
import json
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn import ensemble, metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix

from predict_drug_target.utils import log, TrainingConfig
from predict_drug_target.vectordb import init_vectordb

vectordb = init_vectordb(recreate=False)


def generate_dt_pairs(dt_df):
    """Get pairs and their labels: All given known drug-target pairs are 1,
    We add pairs for missing drug/targets combinations as 0 (not known as interacting)"""
    dtKnown = {tuple(x) for x in dt_df[["drug", "target"]].values}
    pairs = []
    labels = []

    drugs = set(dt_df.drug.unique())
    targets = set(dt_df.target.unique())
    for d in drugs:
        for t in targets:
            label = 1 if (d, t) in dtKnown else 0

            pairs.append((d, t))
            labels.append(label)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        score = scorer(estimator, X_test) if y_test is None else scorer(estimator, X_test, y_test)

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
                f"scoring must return a number, got {score!s} ({type(score)}) " f"instead. (scorer={name})"
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
    print(f"Features count: {len(features_cols)}")
    X = train_df[features_cols].values
    y = train_df["Class"].values.ravel()

    X_new = test_df[features_cols].values
    y_new = test_df["Class"].values.ravel()

    print("FIT X Y")
    print(X)
    print(y)

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


def cv_distribute(run_index, pairs, classes, cv, embedding_df, clfs):
    all_scores = pd.DataFrame()
    for fold in cv:
        scores = cv_run(run_index, pairs, classes, embedding_df, fold[0], fold[1], fold[2], clfs)
        all_scores = pd.concat([all_scores, scores], ignore_index=True)

    return all_scores


def kfold_cv(pairs_all, classes_all, embedding_df, clfs, n_run, n_fold, n_proportion, n_seed):
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

        scores = cv_distribute(r, pairs_classes[0], pairs_classes[1], cv_list, embedding_df, clfs)
        scores_df = pd.concat([scores_df, scores], ignore_index=True)
    return scores_df


###### Main training function


def train(
    df_known_interactions: pd.DataFrame,
    df_drugs_embeddings: pd.DataFrame,
    df_targets_embeddings: pd.DataFrame,
    save_model: str = "models/drug_target.pkl",
    config: TrainingConfig | None = None
):
    """Training takes 3 dataframes as input, ideally use CURIEs for drug/target IDs:
    1. a df with known drug-target interactions (2 cols: drug, target)
    2. a df with drug embeddings: drug col + 512 cols for embeddings
    3. a df with target embeddings: target col + 1280 cols for embeddings
    """
    if not config:
        config = TrainingConfig()

    embeddings = {
        "drug": df_drugs_embeddings,
        "target": df_targets_embeddings,
    }

    today = date.today()
    results_file = f"./data/results/drugbank_drug_targets_scores_{today}.csv"
    agg_results_file = f"./data/results/drugbank_drug_targets_agg_{today}.csv"

    # Get pairs
    pairs, labels = generate_dt_pairs(df_known_interactions)
    ndrugs = len(embeddings["drug"])
    ntargets = len(embeddings["target"])
    unique, counts = np.unique(labels, return_counts=True)
    ndrugtargets = counts[1]
    log.info(f"Training based on {ndrugtargets} Drug-Targets known interactions: {ndrugs} drugs | {ntargets} targets")

    # nb_model = GaussianNB()
    # lr_model = linear_model.LogisticRegression()
    # rf_model = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
    # rf_model = ensemble.RandomForestClassifier(
    #     n_estimators=200,
    #     criterion="log_loss",
    #     max_depth=config.max_depth,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features="sqrt",
    #     n_jobs=-1,
    # )
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=config.max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective='binary:logistic',  # For binary classification
        n_jobs=-1,
        random_state=42,
        tree_method='hist', # Use GPU optimized histogram algorithm
        # device='gpu',
    )

    # clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model)]
    clfs = [("XGBoost", xgb_model)] # "Random Forest", rf_model

    n_seed = 100
    n_fold = config.cv_nfold
    n_run = 2
    n_proportion = 1

    # Run training
    all_scores_df = kfold_cv(pairs, labels, embeddings, clfs, n_run, n_fold, n_proportion, n_seed)
    all_scores_df.to_csv(results_file, sep=",", index=False)

    agg_df = all_scores_df.groupby(["method", "run"]).mean().groupby("method").mean()
    agg_df.to_csv(agg_results_file, sep=",", index=False)
    log.info("Aggregated results:")
    print(agg_df)

    os.makedirs("models", exist_ok=True)
    with open(save_model, "wb") as f:
        pickle.dump(xgb_model, f) #rf_model

    return agg_df
    # return agg_df.to_dict(orient="records")


################### Train with a grid of hyperparameters to find the best


# def get_params_combinations(params):
# 	keys, values = zip(*params.items())
# 	combinations = [dict(zip(keys, v)) for v in product(*values)]
# 	return combinations

def train_gpu(
    df_known_interactions: pd.DataFrame,
    df_drugs_embeddings: pd.DataFrame,
    df_targets_embeddings: pd.DataFrame,
    params: dict[str, int | float],
    save_model: str = "models/drug_target.pkl",
):
    """Train and compare a grid of hyperparameters

    Training takes 3 dataframes as input, ideally use CURIEs for drug/target IDs:
    1. a df with known drug-target interactions (2 cols: drug, target)
    2. a df with drug embeddings: drug col + 512 cols for embeddings
    3. a df with target embeddings: target col + 1280 cols for embeddings
    """
    time_start = datetime.now()
    embeddings = {
        "drug": df_drugs_embeddings,
        "target": df_targets_embeddings,
    }

    log.info("Generate Drug-Target pairs DF")
    # Get pairs and their labels: All given known drug-target pairs are 1
    # we add pairs for missing drug/targets combinations as 0 (not known as interacting)
    pairs, labels = generate_dt_pairs(df_known_interactions)

    log.info("Merging drug/target labels to the DF")
    # Merge drug/target pairs and their labels in a DF
    train_df = pd.DataFrame(
        list(zip(pairs[:, 0], pairs[:, 1], labels)), columns=["drug", "target", "Class"]
    )
    log.info("Merging embeddings to the DF")
    # Add the embeddings to the DF
    train_df = train_df.merge(embeddings["drug"], left_on="drug", right_on="drug").merge(
        embeddings["target"], left_on="target", right_on="target"
    )

    log.info("Getting X and y data")
    # X is the array of embeddings (drug+target), without other columns
    # y is the array of classes/labels (0 or 1)
    embedding_cols = train_df.columns.difference(["drug", "target", "Class"])
    X = train_df[embedding_cols].values
    y = train_df["Class"].values.ravel()
    log.info(f"Features count: {len(embedding_cols)}")
    # print(X)
    # print(y)

    ndrugs = len(embeddings["drug"])
    ntargets = len(embeddings["target"])
    _unique, counts = np.unique(labels, return_counts=True)
    ndrugtargets = counts[1]
    log.info(f"Training based on {ndrugtargets} Drug-Targets known interactions: {ndrugs} drugs | {ntargets} targets")
    random_state=123 # Or 42?
    n_jobs = 2 # Or -1
    n_splits = 5

    # pairs, classes = balance_data(pairs_all, classes_all, n_proportion)
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # TODO: xgboost don't support gridsearch on GPU by default
    # https://github.com/compomics/ms2pip/blob/a8c61b41044f3f756b4551d7866d8030e68b1570/train_scripts/train_xgboost_c.py#L143

    # NOTE: To run XGB on GPU:
    params["device"] = "cuda:0"
    params["tree_method"] = "hist"

    # combinations = get_params_combinations(params_grid)
    # print("Working on combination {}/{}".format(count, len(combinations)))
    combination_time = time.time()
    fold_results = []
    best_accuracy = 0
    os.makedirs("models", exist_ok=True)

    # for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Train model for each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Send data to GPU for XGBoost
        send_time = time.time()
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        # print(f"Sending data to GPU took {time.time() - send_time}s")

        # Train XGBoost model
        model = xgb.train(params, dtrain, num_boost_round=100)
        predictions = model.predict(dtest)

        # Train Random Forest model
        # model = RandomForestClassifier(**params)
        # model.fit(x_train, y_train)
        # predictions = model.predict(x_test)

        # Evaluate model
        predictions_binary = np.round(predictions) # Convert probabilities to binary outputs

        # Calculate metrics
        rmse = np.sqrt(((predictions - y_test) ** 2).mean())
        precision = precision_score(y_test, predictions_binary)
        recall = recall_score(y_test, predictions_binary)
        accuracy = accuracy_score(y_test, predictions_binary)
        roc_auc = roc_auc_score(y_test, predictions)
        f1 = f1_score(y_test, predictions_binary)
        average_precision = average_precision_score(y_test, predictions)

        fold_results.append({
            'rmse': rmse,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'average_precision': average_precision
        })

        # Check if model is better than others, and dump the model in a local file if it is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            with open(save_model, "wb") as f:
                pickle.dump(model, f)

        # os.makedirs("models", exist_ok=True)
        # with open(save_model_path, "wb") as f:
        #     pickle.dump(model, f)

        # rmse = np.sqrt(((predictions - y_test) ** 2).mean())
        # fold_results.append(rmse)
        # del dtrain, dtest, model
        gc.collect()  # Force garbage collection for xgb on GPU
        print(fold_results)
        log.info(f"Completed fold {fold + 1}/{n_splits} in {time.time() - send_time}s")

    log.info(f"Combination took {time.time() - combination_time}s")

    # # Store the average RMSE for this parameter combination
    # avg_rmse = np.mean(fold_results)
    # results.append({'rmse': avg_rmse})
    # # count += 1
    # df = pd.DataFrame(results)

    df_avg_metrics = pd.DataFrame(fold_results).mean()
    print("TRAINING RESULTS")
    print(df_avg_metrics)
    return df_avg_metrics




################## Functions to exclude drugs/targets that are too similar

def drop_similar(df: str, col_id: str, threshold: float = 0.9):
    """Given a DF remove all entities that are too similar"""
    vectordb = init_vectordb(recreate=False)
    indices_to_drop = []
    # TODO: remove things that are too similar
    # in df_drugs and df_targets
    for i, row in df.iterrows():
        if row[col_id] in indices_to_drop:
            # If we already plan to drop this row, skip it
            continue
        # The column ID and the collection are the same (drug or target)
        ent_matching = vectordb.get(col_id, row[col_id])
        if ent_matching:
            # Find vectors that are similar to the vector of the given drug ID
            search_res = vectordb.search(col_id, ent_matching[0].vector)
            for res in search_res:
                if threshold < res.score < 1:
                    indices_to_drop.append(res.payload['id'])
                    df = df[df[col_id] != res.payload['id']]
                # print(f"{res.payload['id']}: {res.score} ({res.id})")
        else:
            print(f"No match for {row[col_id]}")
    log.info(f"DROPPING {col_id}: {len(indices_to_drop)}")
    # return df.drop(indices_to_drop)
    return df



def exclude_similar(input_dir, subject_sim_threshold: float = 1, object_sim_threshold: float = 1):
    """Exclude similarities given thresholds, and run training on grid"""

    print(f"🔨 Training for {subject_sim_threshold} - {object_sim_threshold}")

    # Precomputed embeddings
    df_known_dt = pd.read_csv(f"{input_dir}/known_drugs_targets.csv")
    df_drugs = pd.read_csv(f"{input_dir}/drugs_embeddings.csv")
    df_targets = pd.read_csv(f"{input_dir}/targets_embeddings.csv")

    log.info(f"DF LENGTH BEFORE DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    if subject_sim_threshold < 1:
        df_drugs = drop_similar(df_drugs, "drug", subject_sim_threshold)

    if object_sim_threshold < 1:
        df_targets = drop_similar(df_targets, "target", object_sim_threshold)

    # TODO: remove drugs/targets for which we don't have smiles/AA seq?
    df_known_dt = df_known_dt[df_known_dt['drug'].isin(df_drugs['drug']) & df_known_dt['target'].isin(df_targets['target'])]

    log.info(f"DF LENGTH AFTER DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")
    print(df_known_dt)
    return df_known_dt, df_drugs, df_targets


if __name__ == "__main__":
    # train_grid_exclude_sim("data/opentargets", "data/grid")
    # train_not_similar("data/opentargets", "data/opentargets_not_similar")
    out_dir = "data/grid"
    os.makedirs(out_dir, exist_ok=True)

    # Longer version:
    # subject_sim_thresholds = [1, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    # object_sim_thresholds = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
    subject_sim_thresholds = [1]
    object_sim_thresholds = [1]
    params = {
        'max_depth': 3,
        'n_estimators': 100,
        # For XGB:
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
    }
    scores_df = pd.DataFrame()
    for subject_sim_threshold in subject_sim_thresholds:
        for object_sim_threshold in object_sim_thresholds:
            # Exclude similar then run training on GPU
            df_known_dt, df_drugs_embeddings, df_targets_embeddings  = exclude_similar("data/opentargets", subject_sim_threshold, object_sim_threshold)
            print(f"Similar excluded for {subject_sim_threshold}/{object_sim_threshold}")

            scores = train_gpu(df_known_dt, df_drugs_embeddings, df_targets_embeddings, params)
            scores["subject_sim_threshold"] = subject_sim_threshold
            scores["object_sim_threshold"] = object_sim_threshold
            scores_df = pd.concat([scores_df, scores], ignore_index=True)

    print("SCORES DF", scores_df)
    scores_df.to_csv(f"{out_dir}/compare_scores.csv", index=False)
