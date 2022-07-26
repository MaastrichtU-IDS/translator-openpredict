#!/bin/bash

#git add .
#git commit -m "Improvements"
#git push

ssh ids2 'cd /data/deploy-services/translator-openpredict ; git pull ; docker-compose -f docker-compose.yml -f docker-compose.prod.yml build ; docker-compose down ; docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate'
