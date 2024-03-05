import json
import requests
import glob
import os
from tqdm import tqdm
import csv
import pandas as pd

def get_jsonl_files(target_directory):
    """Return a list of JSONL files from the target directory."""
    return glob.glob(os.path.join(target_directory, '*.json'))

def extract_data_from_jsonl(filename):
    """Extract drugId and targetId from a JSONL file."""
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            yield data.get("drugId", None), data.get("targetId", None)

def get_smiles_from_drugId(drugId):
    """Retrieve the SMILES string for a given drugId."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{drugId}?format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            smiles = data['molecule_structures']['canonical_smiles']
            return smiles
        except Exception as e:
            mol_type = data['molecule_type']
            #print(f"No smiles for drug {drugId} - {mol_type}")
            return None

    return None

def get_sequence_from_targetId(targetId):
    """Retrieve the sequence for a given targetId."""
    url = f"https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:{targetId}?offset=0&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            return data[0].get('sequence', {}).get('sequence', None)
        except Exception as e:
            print(f"No result obtained for {targetId}")
            return None
    return None

def write_to_csv(output_csv, data, header=None):
    """Write the provided data to a CSV file."""
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if header:
            csv_writer.writerow(header)
        if isinstance(data, list):
            for row in data:
                csv_writer.writerow(row)
        elif isinstance(data, dict):
            for row in data.values():
                csv_writer.writerow(row.values())

def main(target_directory, output_directory):
    """Main function to orchestrate the extraction and saving process."""
    drug_targets = []

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(target_directory)
    for json_file in tqdm(json_files, desc="Processing files"):
        for drugId, targetId in extract_data_from_jsonl(json_file):
            drug_targets.append((drugId, targetId))

    write_to_csv(f'{output_directory}/opentargets_drug_targets.csv',
                    drug_targets, ['DrugId', 'TargetId'])

    # create a unique list of drug ids and target ids
    drug_ids = list(set([t[0] for t in drug_targets]))
    target_ids = list(set([t[1] for t in drug_targets]))

    # now retrieve smiles and sequences
    hashes = {}
    file = f'{output_directory}/opentargets_drugs.csv'
    if os.path.exists(file):
       with open(file, 'r') as csvfile:
            df = pd.read_csv(csvfile)
            hashes = df.to_dict('index')
    invalid_smiles = []
    for drugId in tqdm(drug_ids, desc="Processing drugs"):
        if drugId in df['drug_id'].values \
            or drugId in invalid_smiles:
            continue

        smiles = get_smiles_from_drugId(drugId)
        if smiles is None:
            invalid_smiles.append(drugId)
            continue

        h = hash(smiles)
        if h in hashes:
            # retrieve the object
            o = hashes[h]
            if drugId not in o['other_ids']:
                o['other_ids'].append(drugId)
            hashes[h] = o
        else:
            o = {}
            o['hash'] = h
            o['drug_id'] = drugId
            o['smiles'] = smiles
            o['all_ids'] = [drugId]
            df.add()
            hashes[h] = o
    write_to_csv(file, hashes,
                  ['hash','drug_id','smiles','all_ids'])
    write_to_csv(f'{output_directory}/opentargets_no_smiles4drugs.csv', invalid_smiles,
                  ['drug_id'])

    hashes = {}
    for targetId in tqdm(target_ids, desc="Processing targets"):
        sequence = get_sequence_from_targetId(targetId)
        h = hash(sequence)
        if h in hashes:
            # retrieve the object
            o = hashes[h]
            if targetId not in o['other_ids']:
                o['other_ids'].append(targetId)
            hashes[h] = o
        else:
            o = {}
            o['hash'] = h
            o['target_id'] = targetId
            o['sequence'] = sequence
            o['all_ids'] = [targetId]
            hashes[h] = o
    hashes
    write_to_csv(f'{output_directory}/opentargets_targets.csv',
                 hashes, ['hash','target_id','sequence','all_ids'])

if __name__ == "__main__":
    target_directory = 'data/download/opentargets/knownDrugsAggregated'
    output_directory = "data/processed"  # Replace with desired output CSV file name/path
    main(target_directory, output_directory)
