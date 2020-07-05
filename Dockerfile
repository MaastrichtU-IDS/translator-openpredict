FROM python:3.7

# RUN pip install --upgrade pip

COPY . .

RUN pip install -e .

EXPOSE 8808

ENTRYPOINT [ "openpredict" ]
CMD [ "start-api" ]