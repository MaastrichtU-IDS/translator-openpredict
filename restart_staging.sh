#!/bin/bash

#git add .
#git commit -m "Improvements"
#git push

ssh ids3 'cd /data/deploy-services/translator-openpredict ; git pull ; docker-compose -f docker-compose.yml -f docker-compose.staging.yml build ; docker-compose down ; docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d --force-recreate'
