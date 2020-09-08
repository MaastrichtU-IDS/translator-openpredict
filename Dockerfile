FROM jupyter/all-spark-notebook
# FROM jupyter/pyspark-notebook
# Without Spark: FROM python:3.7 

COPY . .

# Install from source code
RUN pip install .

EXPOSE 8808

ENTRYPOINT [ "openpredict" ]
CMD [ "start-api" ]
