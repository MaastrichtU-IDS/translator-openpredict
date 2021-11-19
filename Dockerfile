FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
## Gunicorn image 3.39GB: https://github.com/tiangolo/uvicorn-gunicorn-docker/tree/master/docker-images

# FROM jupyter/pyspark-notebook:python-3.8.8
## Jupyter with Spark already installed 4.31GB

LABEL org.opencontainers.image.source="https://github.com/MaastrichtU-IDS/translator-openpredict"

## Change the current user to root and the working directory to /app
USER root
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential curl vim openjdk-11-jdk wget

## Install Spark for standalone context in /opt
ENV APACHE_SPARK_VERSION=3.2.0
ENV HADOOP_VERSION=3.2
ENV SPARK_HOME=/opt/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx2048M --driver-java-options=-Dlog4j.logLevel=info"
ENV PATH="${PATH}:${SPARK_HOME}/bin"
RUN wget -q -O spark.tgz https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark.tgz -C /opt && \
    rm "spark.tgz" && \
    ln -s "/opt/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" $SPARK_HOME


## Define some environment variables
ENV OPENPREDICT_DATA_DIR=/data/openpredict
ENV PYSPARK_PYTHON=/usr/local/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3
# ENV PYSPARK_PYTHON=/opt/conda/bin/python3
# ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python3

# Avoid to reinstall packages when no changes to requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG INSTALL_DEV=false
COPY requirements-dev.txt .
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then pip install -r requirements-dev.txt ; fi"

## Copy the source code (in the same folder as the Dockerfile)
COPY . .

## Gunicorn config
ENV MODULE_NAME=openpredict.main
ENV VARIABLE_NAME=app
ENV PORT=8808

## Install the pip package based on the source code

RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then pip install -e . ; else pip install . ; fi"

EXPOSE 8808

# ENTRYPOINT ["uvicorn", "openpredict.main:app",  "--host", "0.0.0.0", "--port", "8808"]
# ENTRYPOINT [ "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8808", "openpredict.main:app"]
# ENTRYPOINT [ "openpredict", "start-api" ]
