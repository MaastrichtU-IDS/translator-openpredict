FROM jupyter/all-spark-notebook:spark-3.1.1
# FROM jupyter/pyspark-notebook
# Without Spark: FROM python:3.7 

# Required to be able to edit the .joblib model directly in the python package
USER root
WORKDIR /root

ENV OPENPREDICT_DATA_DIR=/data/openpredict
ENV PYSPARK_PYTHON=/opt/conda/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python3

RUN apt-get update && apt-get install -y build-essential

COPY . .

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER $NB_USER

# Install from source code
RUN pip install .

EXPOSE 8808
ENTRYPOINT [ "openpredict", "start-api" ]