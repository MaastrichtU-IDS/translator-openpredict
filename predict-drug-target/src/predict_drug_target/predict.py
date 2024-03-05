import pickle
from datetime import datetime

import pandas as pd
from trapi_predict_kit import PredictInput, PredictOutput, trapi_predict

from predict_drug_target.embeddings import compute_drug_embedding, compute_target_embedding
from predict_drug_target.utils import (
    BOLD,
    COLLECTIONS,
    END,
    log,
)
from predict_drug_target.vectordb import init_vectordb

VECTORDB = init_vectordb(recreate=False)


def load_model(path: str = "models/drug_target.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


@trapi_predict(
    path="/predict-drug-target",
    name="Get predicted score for interactions between drugs and targets (protein)",
    description="Return the predicted targets for a given entity: drug (PubChem ID) or target (UniProtKB ID), with confidence scores.",
    edges=[
        {
            "subject": "biolink:Drug",
            "predicate": "biolink:interacts_with",
            "inverse": "biolink:interacts_with",
            "object": "biolink:Protein",
        },
    ],
    nodes={
        "biolink:Protein": {"id_prefixes": ["UniProtKB"]},
        "biolink:Drug": {"id_prefixes": ["PUBCHEM.COMPOUND", "CHEMBL.COMPOUND"]},
    },
)
def get_drug_target_predictions(request: PredictInput) -> PredictOutput:
    time_start = datetime.now()
    model = load_model()

    # Compute embeddings for drugs and target, based on their smiles and amino acid sequence
    drug_embed = compute_drug_embedding(VECTORDB, request.subjects)
    target_embed = compute_target_embedding(VECTORDB, request.objects)
    # print("DRUGS TARGETS", drug_embed)
    # print(target_embed)

    # Merge embeddings, results should have 1792 columns (512 from drugs + 1280 from targets)
    df = pd.merge(drug_embed, target_embed, how="cross")
    df.columns = df.columns.astype(str)
    merged_embeddings = df.drop(columns=["drug", "target"])
    merged_embeddings.columns = range(merged_embeddings.shape[1])  # use default column names, same as during training
    # log.info(df)

    # Get predicted score
    predicted_proba = model.predict_proba(merged_embeddings)
    df["score"] = predicted_proba[:, 1]  # Probability of class 1
    df = df.sort_values(by="score", ascending=False)
    df.rename(columns={"drug": "subject", "target": "object"}, inplace=True)
    score_df = df[["subject", "object", "score"]]
    # Convert to list of dicts
    log.info(
        f"⚡ {BOLD}{len(df)}{END} interaction scores computed in {BOLD}{datetime.now() - time_start}{END}\n{score_df.iloc[:10]}"
    )
    scores_list = score_df.to_dict(orient="records")
    return {"hits": scores_list, "count": len(scores_list)}
