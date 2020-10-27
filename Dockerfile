FROM jupyter/all-spark-notebook
# FROM jupyter/pyspark-notebook
# Without Spark: FROM python:3.7 

# Required to be able to edit the .joblib model directly in the python package
USER root

RUN apt-get update && apt-get install -y build-essential

COPY . .

# Install from source code
RUN pip install .

EXPOSE 8808

ENTRYPOINT [ "openpredict" ]
CMD [ "start-api" ]
