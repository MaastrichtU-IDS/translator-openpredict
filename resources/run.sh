#!/usr/bin/env bash

set -e


if [ $# -eq 0 ]
then
    echo "No arguments supplied, defaulting to 'python src/openpredict_model/train.py train-model'"
    args="python src/openpredict_model/train.py train"
else
    args=$@
fi


# scripts/run.sh python src/openpredict_model/train.py train-model

docker-compose run --entrypoint "$args" tests
