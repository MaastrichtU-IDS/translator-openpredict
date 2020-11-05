# Reset models, and metadata to the last git commit

git checkout HEAD -- openpredict/data/openpredict-metadata.ttl
git checkout HEAD -- openpredict/data/features/openpredict-baseline-omim-drugbank.joblib
git checkout HEAD -- openpredict/data/models/openpredict-baseline-omim-drugbank.joblib