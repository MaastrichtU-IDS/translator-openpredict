#!/bin/bash

ssh ids3 'cd /data/deploy-services/qdrant-vectordb ; git pull ; docker-compose up --force-recreate --build -d'

# ssh ids3 'cd /data/deploy-services/predict-drug-target/vectordb ; git pull ; docker-compose up --force-recreate --build -d'
