ARG BASE_IMAGE=tiangolo/uvicorn-gunicorn-fastapi:python3.10
FROM ${BASE_IMAGE}
# Gunicorn image 3.4G: https://github.com/tiangolo/uvicorn-gunicorn-docker/tree/master/docker-images


LABEL org.opencontainers.image.source="https://github.com/MaastrichtU-IDS/translator-openpredict"

# Change the current user to root and the working directory to /app
USER root
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential wget curl vim && \
    pip install --upgrade pip

# RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

ENV PORT=8808 \
    GUNICORN_CMD_ARGS="--preload" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACCESS_LOG="-" \
    ERROR_LOG="-" \
    OPENTELEMETRY_ENABLED=false

# # Use requirements.txt to install some dependencies only when needed
# COPY trapi-openpredict/requirements.txt requirements.txt
# RUN pip install -r requirements.txt

## Copy the source code (in the same folder as the Dockerfile)
COPY . .

ENV MODULE_NAME=src.oprenpredict_trapi.main \
    VARIABLE_NAME=app

WORKDIR /app/trapi-openpredict

# RUN pip install -e /app/predict-drug-target /app/trapi-predict-kit
RUN pip install -e .

# RUN pip install -e . /app/predict-drug-target /app/trapi-predict-kit
# RUN pip install -e /app/trapi-predict-kit

RUN dvc pull -f

EXPOSE 8808

# ENTRYPOINT [ "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8808", "trapi.main:app"]

# Build entrypoint script to pull latest dvc changes before startup
RUN echo "#!/bin/bash" > /entrypoint.sh && \
    echo "dvc pull" >> /entrypoint.sh && \
    echo "/start.sh" >> /entrypoint.sh && \
    chmod +x /entrypoint.sh


CMD [ "/entrypoint.sh" ]
