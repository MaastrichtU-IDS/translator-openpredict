FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
# Gunicorn image 3.4G: https://github.com/tiangolo/uvicorn-gunicorn-docker/tree/master/docker-images


LABEL org.opencontainers.image.source="https://github.com/MaastrichtU-IDS/translator-openpredict"

# Change the current user to root and the working directory to /app
USER root
WORKDIR /app


RUN apt-get update && \
    apt-get install -y build-essential wget curl vim openjdk-11-jdk && \
    pip install --upgrade pip

# Install Poetry
# ENV POETRY_VERSION=1.2.0
# RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION
# ENV PATH=/root/.local/bin:$PATH
# RUN poetry config virtualenvs.create false



# Install Spark for standalone context in /opt
ENV APACHE_SPARK_VERSION=3.2.0
ENV HADOOP_VERSION=3.2
ENV SPARK_HOME=/opt/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx2048M --driver-java-options=-Dlog4j.logLevel=info"
ENV PATH="${PATH}:${SPARK_HOME}/bin"
RUN wget -q -O spark.tgz https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark.tgz -C /opt && \
    rm "spark.tgz" && \
    ln -s "/opt/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" $SPARK_HOME
RUN echo "log4j.rootCategory=ERROR, console" > $SPARK_HOME/conf/log4j.properties
# RUN chown -R 1000:1000 /opt/spark

# Define some environment variables for pyspark and gunicorn config
ENV PYSPARK_PYTHON=/usr/local/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3
ENV MODULE_NAME=trapi.main
ENV VARIABLE_NAME=app
ENV PORT=8808
ENV GUNICORN_CMD_ARGS="--preload"

# Use requirements.txt to install some dependencies only when needed
COPY requirements.txt .
RUN pip install -r requirements.txt

## Copy the source code (in the same folder as the Dockerfile)
COPY . .

RUN pip install -e ".[train,test,dev]"

RUN dvc pull -f

EXPOSE 8808

# ENTRYPOINT [ "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8808", "trapi.main:app"]

# Build entrypoint script to pull latest dvc changes before startup
RUN echo "#!/bin/bash" > /entrypoint.sh && \
    echo "dvc pull" >> /entrypoint.sh && \
    echo "/start.sh" >> /entrypoint.sh && \
    chmod +x /entrypoint.sh


CMD [ "/entrypoint.sh" ]
