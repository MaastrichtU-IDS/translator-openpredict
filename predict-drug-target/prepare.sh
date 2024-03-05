#!/usr/bin/bash
# pip install git+https://github.com/facebookresearch/esm.git

# Get drug-target data from the Bio2RDF SPARQL endpoint
./get_bio2rdf_data.sh

# Generate drugs embeddings
esm-extract esm2_t33_650M_UR50D data/download/drugbank_targets.fasta data/vectors/drugbank_targets_esm2_l33_mean --repr_layers 33 --include mean


# Clustering sim with mmseq
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH

mmseqs easy-cluster data/DB.fasta clusterRes tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
mmseqs createtsv clu_rep clu_rep clusterRes.tsv

mmseqs createtsv sequenceDB sequenceDB resultsDB_clu resultsDB_clu.tsv



# Install the Molecular Transformer Embeddings for proteins
# https://github.com/mpcrlab/MolecularTransformerEmbeddings
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
cd MolecularTransformerEmbeddings
chmod +x download.sh
./download.sh
python embed.py --data_path=../data/download/drugbank_smiles.txt
mkdir -p ../data/vectors
mv embeddings/drugbank_smiles.npz ../data/vectors/
cd ..


echo "Generate list of known_drug_target pairs for OpenTargets"
python3 src/prepare.py
