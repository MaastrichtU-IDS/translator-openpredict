FROM jupyter/all-spark-notebook
# FROM jupyter/pyspark-notebook
# Without Spark: FROM python:3.7 

# Required to be able to edit the .joblib model directly in the python package
USER root

ENV OPENPREDICT_DATA_DIR=/data/openpredict
ENV PYSPARK_PYTHON=/opt/conda/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python3

RUN apt-get update && apt-get install -y build-essential

COPY . .

# Install from source code
RUN pip install .

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 8808

# ENTRYPOINT [ "sleep 60 && openpredict" ]
# CMD [ "start-api" ]
