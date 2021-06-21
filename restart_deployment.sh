#!/bin/bash

#git add .
#git commit -m "Improvements"
#git push

ssh ids2 'cd /data/deploy-ids-tests/translator-openpredict ; git pull ; docker-compose down ; docker-compose build --no-cache ; docker-compose up -d'
