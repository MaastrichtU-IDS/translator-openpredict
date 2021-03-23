# Delete all files but the initial metadata file in data folder

find ./data/* ! -name \"initial-openpredict-metadata.ttl\" -exec sudo rm -rf {} +
docker-compose up -d --force-recreate


curl -X POST "https://ars.transltr.io/ars/api/submit" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{
  \"message\": {
    \"query_graph\": {
      \"edges\": {
        \"e01\": {
          \"subject\": \"n0\",
          \"predicate\": \"biolink:treats\",
          \"object\": \"n1\"
        }
      },
      \"nodes\": {
        \"n0\": {
          \"category\": \"biolink:Drug\",
          \"id\": \"DRUGBANK:DB00394\"
        },
        \"n1\": {
          \"category\": \"biolink:Disease\"
        }
      }
    }
  }
}"