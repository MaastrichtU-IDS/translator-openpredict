import glob
import json
import os

import pandas as pd
from tqdm import tqdm
from predict_drug_target import vectordb

# from predict_drug_target.embeddings import compute_drug_embedding, compute_target_embedding
from predict_drug_target.embeddings import compute
from predict_drug_target.utils import COLLECTIONS, log, get_pref_ids
from predict_drug_target.vectordb import init_vectordb

# NOTE: script to run the WHOLE pipeline on opentargets data
# it will automatically compute embeddings for all drugs and targets
# Download opentargets before running this script: ./scripts/download_opentargets.sh

# Output file path
# output_file_path = "../data/opentargets/merged_parsed.csv"


def get_jsonl_files(target_directory) -> list[str]:
    """Return a list of JSONL files from the target directory."""
    return glob.glob(os.path.join(target_directory, "*.json"))


def extract_data_from_jsonl(filename):
    """Extract drugId and targetId from a JSONL file."""
    with open(filename) as file:
        for line in file:
            data = json.loads(line.strip())
            yield data.get("drugId", None), data.get("targetId", None)



def ensembl_to_uniprot():
    """Dict to convert ENSEMBL IDs to UniProt IDs"""
    json_files = get_jsonl_files("data/download/opentargets/targets")
    ensembl_to_uniprot_dict = {}

    for json_file in tqdm(json_files, desc="Mapping targets ENSEMBL IDs to UniProt"):
        with open(json_file) as file:
            for line in file:
                data = json.loads(line.strip())
                for prot in data.get("proteinIds", []):
                    if prot["source"] == "uniprot_swissprot":
                        ensembl_to_uniprot_dict[data["id"]] = f"UniProtKB:{prot['id']}"

    return ensembl_to_uniprot_dict


# NOTE: to train the model on new data you will just need a CSV of known drug-target pairs with 2 columns: `drug` and `target`
# Use CURIEs for the drugs and targets IDs. Accepted namespaces: UniProtKB:, PUBCHEM.COMPOUNT:, CHEMBL.COMPOUND:

def prepare_opentargets(input_dir, out_dir):
    """Compute embeddings and train the model using opentargets data."""
    os.makedirs(out_dir, exist_ok=True)
    known_drug_targets = []

    ensembl_to_uniprot_dict = ensembl_to_uniprot()
    no_match = set()
    print(len(ensembl_to_uniprot_dict))

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(input_dir)
    for json_file in tqdm(json_files, desc="Processing files"):
        # log.info(json_file)
        for drug_id, target_id in extract_data_from_jsonl(json_file):
            try:
                known_drug_targets.append(
                    {
                        "drug": f"CHEMBL.COMPOUND:{drug_id}",
                        "target": ensembl_to_uniprot_dict[target_id],
                    }
                )
            except:
                no_match.add(target_id)

    log.info(f"No UniProt match for {len(no_match)} targets, e.g. {' ,'.join(list(no_match))}")

    df_known_dt = pd.DataFrame(known_drug_targets)
    known_dt_path = f"{out_dir}/known_drugs_targets.csv"

    print(df_known_dt)
    print(f"Known drug-targets pairs stored in {known_dt_path}")
    df_known_dt.to_csv(known_dt_path)

    print("Computing embeddings")
    df_known_dt2, df_drugs, df_targets = compute(df_known_dt, init_vectordb(), out_dir)

    df_drugs.to_csv(f"{out_dir}/drugs_embeddings.csv")
    df_targets.to_csv(f"{out_dir}/targets_embeddings.csv")

    # NOTE: block to skip computing
    # df_known_dt, df_drugs, df_targets = compute(df_known_dt, out_dir)
    # df_known_dt = pd.read_csv(f"data/opentargets/known_drugs_targets.csv")
    # df_drugs = pd.read_csv(f"data/opentargets/drugs_embeddings.csv")
    # df_targets = pd.read_csv(f"data/opentargets/targets_embeddings.csv")
    # scores = train(df_known_dt, df_drugs, df_targets, f"{out_dir}/model.pkl")


def prepare_drugbank():
    """Compute embeddings and train the model using drugbank data."""
    file_known_dt = "data/drugbank/DB_DTI_4vectordb.csv"
    out_dir = "data/drugbank"

    df_known_dt = pd.read_csv(file_known_dt)
    # Convert DrugBank IDs to PubChem
    convert_dict = get_pref_ids(df_known_dt["drug"].values, ["PUBCHEM.COMPOUND"])
    print(convert_dict)
    df_known_dt["drug"] = df_known_dt["drug"].apply(lambda curie: convert_dict[curie])
    print(df_known_dt)
    df_known_dt.to_csv("data/drugbank/known_drugs_targets.csv")
    # df_known_dt, df_drugs, df_targets = compute(df_known_dt, out_dir)
    # scores = train(df_known_dt, df_drugs, df_targets, f"{out_dir}/model.pkl")

if __name__ == "__main__":
    # prepare_drugbank()
    prepare_opentargets("data/download/opentargets/knownDrugsAggregated", "data/opentargets")
