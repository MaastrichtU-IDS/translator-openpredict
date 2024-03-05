import logging
from time import sleep
from dataclasses import dataclass

import pubchempy as pcp
import requests

VECTORDB_MAX_LIMIT = 100000

EMBEDDINGS_SIZE_DRUG = 512
EMBEDDINGS_SIZE_TARGET = 1280
COLLECTIONS = [
    {"name": "drug", "size": EMBEDDINGS_SIZE_DRUG},
    {"name": "target", "size": EMBEDDINGS_SIZE_TARGET},
]  # Total 1792 features cols
ACCEPTED_NAMESPACES = ["PUBCHEM.COMPOUND:", "UniProtKB:"]


## Instantiate logging utility
log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(module)s:%(funcName)s] %(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Silence qdrant log.infos
default_logger = logging.getLogger()
default_logger.setLevel(logging.WARNING)


BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

TIMEOUT = 30

@dataclass
class TrainingConfig:
    subject_sim_threshold: int = 1 # 0 to 1
    object_sim_threshold: int = 1
    cv_nfold: int = 10
    max_depth: int = 6



def get_smiles_for_drug(drug_id: str):
    sleep(1) # We cant bulk query, and they fail when too many queries
    # Not all molecule have smiles https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL4297578?format=json
    # CHEMBL.COMPOUND:CHEMBL4297578
    if drug_id.lower().startswith("chembl.compound:"):
        drug_id = drug_id[len("chembl.compound:") :]
        res = requests.get(
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{drug_id}?format=json", timeout=TIMEOUT
        ).json()
        # log.info(f'{drug_id} | {res["molecule_structures"]["canonical_smiles"]} | {res["pref_name"]}')
        return res["molecule_structures"]["canonical_smiles"], res["pref_name"]
    if drug_id.lower().startswith("pubchem.compound:"):
        drug_id = drug_id[len("pubchem.compound:") :]
        comp = pcp.Compound.from_cid(drug_id)
        return comp.canonical_smiles, comp.iupac_name


def get_seq_for_target(target_id: str):
    sleep(1) # We cant bulk query, and they fail when too many queries
    if target_id.lower().startswith("uniprotkb:"):
        target_id = target_id[len("uniprotkb:") :]
        # https://rest.uniprot.org/uniprotkb/B4E0X6?format=json
        res = requests.get(f"https://rest.uniprot.org/uniprotkb/{target_id}?format=json", timeout=TIMEOUT).json()
        return res["sequence"]["value"], res["proteinDescription"]["recommendedName"]["fullName"]["value"]
    # https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:ENSP00000351276?offset=0&size=100&format=json
    # Many possible sequences for 1 Ensembl ID
    # if target_id.lower().startswith("ensembl:"):
    #     target_id = target_id[len("ensembl:") :]
    #     res = requests.get(
    #         f"https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:{target_id}?offset=0&size=100&format=json",
    #         timeout=TIMEOUT,
    #     ).json()
    #     return res[0]["sequence"]["sequence"], res[0]["protein"]["recommendedName"]["fullName"]["value"]


def get_pref_ids(ids_list: list, accepted_namespaces: list[str] = None):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID, a list of accepted namespaces can be passed https://nodenormalization-sri.renci.org/docs
    """
    pref_ids = {}
    resolve_curies = requests.post(
        "https://nodenormalization-sri.renci.org/get_normalized_nodes",
        json={"curies": list(ids_list), "conflate": True, "description": False, "drug_chemical_conflate": False},
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    resolve_curies.raise_for_status()
    resp = resolve_curies.json()
    # print(resp)
    for original_id, available_ids in resp.items():
        pref_id = original_id
        try:
            if not accepted_namespaces:
                pref_id = available_ids["id"]["identifier"]
            else:
                for ns in accepted_namespaces:
                    if available_ids["id"]["identifier"].lower().startswith(ns.lower()):
                        pref_id = available_ids["id"]["identifier"]
                if pref_id == original_id:
                    for alt_id in available_ids["equivalent_identifiers"]:
                        for ns in accepted_namespaces:
                            if alt_id["identifier"].lower().startswith(ns.lower()):
                                pref_id = alt_id["identifier"]
            # log.debug(f"{original_id} > {pref_id}")
        except Exception:
            log.debug(f"Could not find pref ID for {original_id} in {available_ids}")
            # pref_id = original_id
        pref_ids[original_id] = pref_id
    return pref_ids
