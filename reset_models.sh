# Reset models, and metadata to the last git commit

find ./data ! -name 'initial-openpredict-metadata.ttl' -exec sudo rm -rf {} +

# git checkout HEAD -- openpredict/data/openpredict-metadata.ttl
# git checkout HEAD -- openpredict/data/features/openpredict-baseline-omim-drugbank.joblib
# git checkout HEAD -- openpredict/data/models/openpredict-baseline-omim-drugbank.joblib