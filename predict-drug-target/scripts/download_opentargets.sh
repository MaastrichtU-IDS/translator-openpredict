# FTP dont work on UM servers:
# wget --recursive --no-parent --no-host-directories -P ./data/download/opentargets/ --cut-dirs 8 ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated

# HTTP works on UM servers:
wget -r -np -nH --cut-dirs=8 -P ./data/download/opentargets/ -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated/
wget -r -np -nH --cut-dirs=8 -P ./data/download/opentargets/ -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/targets/

# Mechanisms of action:
# wget -r -np -nH --cut-dirs=8 -P ./data/download/opentargets/ -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/mechanismOfAction/
