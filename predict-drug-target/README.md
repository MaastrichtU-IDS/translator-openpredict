# 💊🎯 Predict drug target interactions

This project uses [ESM2](https://github.com/facebookresearch/esm) protein embeddings and [MolecularTransformer](https://github.com/mpcrlab/MolecularTransformerEmbeddings) drug embeddings to train a linear classifier to predict potential drug-targets interactions, where targets are proteins.

Services deployed:

* TRAPI endpoint for drug-target interaction prediction: [predict-drug-target.137.120.31.160.nip.io](https://predict-drug-target.137.120.31.160.nip.io)

*  [Qdrant](https://qdrant.tech/) vector database to store the computed embeddings for drugs and targets: [qdrant.137.120.31.148.nip.io/dashboard](https://qdrant.137.120.31.148.nip.io/dashboard)

## 📥 Install

If you are **not in a docker container** you might want to create and activate local environment before installing the module:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the module:

```bash
pip install -e .
```

## 🍳 Prepare data and embeddings

Run the complete pipeline to download data and generated embeddings for drugs/target:

```bash
./prepare.sh
```


<details><summary>Or click here to follow the pipeline step by step</summary>

Query the Bio2RDF endpoint to get drugs and their smiles, targets and their protein sequences, and the set of known drug-target pairs

```bash
./get_bio2rdf_data.sh
```

Process the Bio2RDF data to generate the inputs needed for the two embeddings methods
```bash
python src/prepare.py
```

Install the ESM library
```bash
pip install git+https://github.com/facebookresearch/esm.git
```

Generate the protein embeddings
```bash
esm-extract esm2_t33_650M_UR50D data/download/drugbank_targets.fasta data/vectors/drugbank_targets_esm2_l33_mean --repr_layers 33 --include mean
```

Install the [Molecular Transformer Embeddings](https://github.com/mpcrlab/MolecularTransformerEmbeddings)
```bash
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
cd MolecularTransformerEmbeddings
chmod +x download.sh
./download.sh
```
if you get an error (bash: ./download.sh: /bin/bash^M: bad interpreter: No such file or directory) running the download script, then run dos2unix

Generate the drug embeddings
```bash
python embed.py --data_path=../data/download/drugbank_smiles.txt
mv embeddings/drugbank_smiles.npz ../data/vectors/
cd ..
```

</details>

## 🏋️ Run training

To force using a specific GPU set the environment variable `CUDA_VISIBLE_DEVICES` (starting from 0, so if you have 3 GPUs you can choose between 0,1 and 2):

```bash
export CUDA_VISIBLE_DEVICES=1
```

Train the model:

```bash
python src/train.py
```

> Results are in the `results/` folder, model pickle goes to the `models/` folder

## 🔮 Get predictions

Run the prediction workflow for 2 entities:

```bash
python src/predict.py
```

Users provides drugs and targets using their CHEMBL or Ensembl IDs, the script will test all provided drugs against all provided targets, and return a prediction score (how confident we are that the drug interacts with the target) for each drug-target pair.
- What takes the most time is computing the embeddings for the drugs and targets, so I started a vector database on one of our server (using the Qdrant vector database), and I am storing the embeddings for the drugs and targets I compute there
- So we try to retrieve the embeddings from the vector database. If not present, we get SMILES and AA sequence for drugs/targets based on their ID using the EBI APIs. Then we compute the embeddings using the same method as for the training (ESM for targets and MolecularTransformerEmbeddings for drugs). And we add those embeddings to the vector database, so we don't need to recompute them next time
- Finally, we merge the drug and target embeddings, and query the model with the merged embeddings to get the interaction probability (between 0 and 1, e.g. 0.45)
- The function currently returns a list of drug/targets pair and their interaction prediction score
- When the embeddings need to be computed it takes ~30s per embeddings. But if the embeddings are already stored in the vector database then the whole prediction process takes less than 2s

## ✅ Run tests

Run the code formatting (black and ruff):

```bash
hatch run fmt
```

Run the tests (requires to first run the training to generate the model):

```bash
pytest
# Or
hatch run test
```

Compile the `requirements.txt` file with pinned versions:
```bash
hatch run requirements
```

## 🐳 Deployment

With docker compose. First run the training to generate the model

### Deploy the API

Deploy the TRAPI endpoint on a GPU server:

```bash
docker compose up -d --build --force-recreate
```

### Deploy the vector db

The vectordb is used to store embeddings for the entities and make querying faster. It is currently hosted on a server.

To run it locally, edit the host in the `src/predict.py` script. And use the `docker-compose.yml` and config files from the `vectordb` folder (make changes as needed)

```bash
cd vectordb
docker compose up -d
```

### Which vector db?

It's the new hot thing in the database world: databases for "modern" AI. To store and query embeddings.

There are a few solutions, more or less mature, here are the runner ups:

* **Qdrant**: the newest kid in the block. It's the fastest according to the few benchmarks I could find. Globally the whole system is really well thought, easy to use, and just fast (simple API, dev/production workflow easy setup, written in rust). Metadata for vectors are stored in a JSON payload.
    * 3 similarity [search algorithms available](https://qdrant.tech/documentation/concepts/search/#metrics): dot, cosine, euclid
* **Milvus**: more features, still quite fast, bug heavier. Has a web UI
    * 3 similarity [search algorithms available](https://milvus.io/docs/metric.md): inner product (is it similar to dot??), cosine, euclid
* **Weaviate**: more battery included, but slower, and heavier. As a GraphQL API (not sure if it's better than a well thought good ol' REST-like API tbh)
    * Similarity search in weaviate seems to be hidden under some additional abstration layers: https://weaviate.io/developers/weaviate/search/similarity, which can make the whole thing harder to curb to our needs. But it seems to also support dot, cosine, and euclid
    * GraphQL API (but is it really needed?)
* **ChromaDB**: more battery included. Does not seems really fast. The code is simplist, all in python, not sure what they are actually bringing (use SQLite as db, parquet for persistence, [similarity search seems to be in memory python](https://github.com/chroma-core/chroma/blob/e81cc9f361e5aa072534a1fbbc483da406b54848/chromadb/segment/impl/vector/local_hnsw.py#L116)).
* **pgvector**: a vertordb in postgres, really nice if you already use SQL or postgres in your system. But for our needs I feel like a simpler NoSQL system would be easier to use and maintain


Some references:
- https://www.brainbyte.io/the-best-vector-databases/
- https://qdrant.tech/benchmarks/
- https://lakefs.io/blog/12-vector-databases-2023/

## ☑️ TODO

- [ ] Store the built model on HuggingFace?
- [ ] Re-train the model on more data from UniProt?
