#!/bin/bash

#git add .
#git commit -m "Improvements"
#git push

ssh ids2 'cd /data/deploy-ids-tests/translator-openpredict ; git pull ; docker-compose build ; docker-compose down ; docker-compose up -d --force-recreate'
