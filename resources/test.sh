#!/usr/bin/env bash

set -e

docker-compose run tests $@
