FROM jupyter/all-spark-notebook
# FROM jupyter/pyspark-notebook
# Without Spark: FROM python:3.7 

# RUN pip install --upgrade pip

COPY . .

# Install from source code
RUN pip install -e .

EXPOSE 8808

ENTRYPOINT [ "openpredict" ]
CMD [ "start-api" ]