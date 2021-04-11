# Delete all files but the initial metadata file in data folder

find ./data/* ! -name 'initial-openpredict-metadata.ttl' -exec sudo rm -rf {} +
docker-compose up -d --force-recreate


