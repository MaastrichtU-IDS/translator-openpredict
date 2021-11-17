FROM jupyter/all-spark-notebook:python-3.8.8
## We use Jupyter to get SPark already installed
## It can be also run with a basic python image: 
# FROM python:3.8 

## Change the current user to root and the working directory to /root
USER root
WORKDIR /root

RUN apt-get update && apt-get install -y build-essential

# RUN fix-permissions $CONDA_DIR && \
#     fix-permissions /home/$NB_USER

# USER $NB_USER

## Define some environment variables
ENV OPENPREDICT_DATA_DIR=/data/openpredict
ENV PYSPARK_PYTHON=/opt/conda/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python3

# Avoid to reinstall packages when no changes to requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

## Copy the source code (in the same folder as the Dockerfile)
COPY . .

## Install the pip package based on the source code
RUN pip install .

## Indicate this will export the port 8808
EXPOSE 8808

ENTRYPOINT [ "openpredict", "start-api" ]

# ENTRYPOINT ["uvicorn", "api.main:app",  "--host", "0.0.0.0", "--port", "8808"]