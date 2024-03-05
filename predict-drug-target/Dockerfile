ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}
# https://ngc.nvidia.com/catalog/containers/nvidia:cuda

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam \
    PYTHONUNBUFFERED=1 \
    JOBLIB_TEMP_FOLDER=/app/data/tmp
    # ^fixes no space left on device error

WORKDIR /app

# CUDA image required to install python
RUN apt-get update && \
    apt-get install -y vim python3-dev python3-venv python3-pip curl wget unzip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # wget https://bootstrap.pypa.io/get-pip.py && \
    # python3 get-pip.py && \
    # rm get-pip.py && \
    pip3 install --upgrade pip

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD . .
RUN pip3 install -e .

CMD [ "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "4", "src.api:app" ]
