# Reset models, and metadata to the last git commit

git checkout HEAD -- openpredict/data/openpredict-metadata.ttl
git checkout HEAD -- openpredict/data/features/drug_disease_dataframes.joblib
git checkout HEAD -- openpredict/data/models/openpredict-baseline-omim-drugbank.joblib