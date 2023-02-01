Additionally to the library, there is an example API, available at **[openpredict.semanticscience.org](https://openpredict.semanticscience.org)**, for drug/disease predictions generated from the OpenPredict model.

The project is structured as follow:

* the code for the OpenPredict API endpoints in  `src/trapi/` defines:
    * a TRAPI endpoint returning predictions for the loaded models
    * individual endpoints for each loaded models, taking an input id, and returning predicted hits
    * endpoints serving metadata about runs, models evaluations, features for the OpenPredict model, stored as RDF, using the [ML Schema ontology](http://ml-schema.github.io/documentation/ML%20Schema.html).
* various folders for **different prediction models** served by the OpenPredict API are available under `src/`:
    * the OpenPredict drug-disease prediction model in `src/openpredict_model/`
    * a model to compile the evidence path between a drug and a disease explaining the predictions of the OpenPredict model in `src/openpredict_evidence_path/`
    * a prediction model trained from the Drug Repurposing Knowledge Graph (aka. DRKG) in `src/drkg_model/`

The data used by the models in this repository is versionned using `dvc` in the `data/` folder, and stored **on DagsHub at [dagshub.com/vemonet/translator-openpredict](https://dagshub.com/vemonet/translator-openpredict)**
