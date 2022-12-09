#!/bin/bash

ssh ids2 'cd /data/deploy-services/translator-openpredict ; git pull ; docker-compose -f docker-compose.prod.yml build ; docker-compose down ; docker-compose -f docker-compose.prod.yml up -d --force-recreate'
