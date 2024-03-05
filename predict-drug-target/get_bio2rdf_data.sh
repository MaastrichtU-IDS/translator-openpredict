curl -F 'format=text/csv' -F "query=@data/queries/drugbank_drug_targets.rq" "https://bio2rdf.org/sparql" > data/download/drugbank_drug_targets.csv
curl -F 'format=text/csv' -F "query=@data/queries/drugbank_drugs.rq" "https://bio2rdf.org/sparql" > data/download/drugbank_drugs.csv
curl -F 'format=text/csv' -F "query=@data/queries/drugbank_targets.rq" "https://bio2rdf.org/sparql" > data/download/drugbank_targets.csv
