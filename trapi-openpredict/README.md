# 🔮🐍 OpenPredict TRAPI endpoint

This repository contains the code for the **OpenPredict Translator API** available at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**, which serves a few prediction models developed at the Institute of Data Science.

* various folders for **different prediction models** served by the OpenPredict API are available under `src/`:
  * the OpenPredict drug-disease prediction model in `src/openpredict_model/`
  * a model to compile the evidence path between a drug and a disease explaining the predictions of the OpenPredict model in `src/openpredict_evidence_path/`
  * a prediction model trained from the Drug Repurposing Knowledge Graph (aka. DRKG) in `src/drkg_model/`
* the code for the OpenPredict API endpoints in  `src/trapi/` defines:
  *  a TRAPI endpoint returning predictions for the loaded models

The data used by the models in this repository is versionned using `dvc` in the `data/` folder, and stored **on DagsHub at https://dagshub.com/vemonet/translator-openpredict**

### Deploy the OpenPredict API locally

Requirements: Hatch to manage python environments

Go this project folder:

```bash
cd trapi-openpredict
```

Make sure local dependencies are up to date:

```bash
hatch run pip install -e ../predict-drug-target ../trapi-predict-kit .
```

### With docker

Start the API in development mode with docker on http://localhost:8808, the API will automatically reload when you make changes in the code:

```bash
docker compose up api
```
