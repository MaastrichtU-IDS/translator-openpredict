## Deploy the OpenPredict API locally

Requirements: Python 3.8+ and `pip` installed

1. Clone the repository:

   ```bash
   git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
   cd translator-openpredict
   ```

2. Pull the data required to run the models in the `data` folder with [`dvc`](https://dvc.org/):

   ```bash
   pip install dvc
   dvc pull
   ```


Start the API in development mode with docker on http://localhost:8808, the API will automatically reload when you make changes in the code:

```bash
docker-compose up api
# Or with the helper script:
./scripts/api.sh
```

> Contributions are welcome! If you wish to help improve OpenPredict, see the [instructions to contribute :woman_technologist:](/CONTRIBUTING.md) for more details on the development workflow

## Test the OpenPredict API

Run the tests locally with docker:

```bash
docker-compose run tests
# Or with the helper script:
./scripts/test.sh
```

> See the [`TESTING.md`](/TESTING.md) file for more details on testing the API.

You can change the entrypoint of the test container to run other commands, such as training a model:

```bash
docker-compose run --entrypoint "python src/openpredict_model/train.py train-model" tests
# Or with the helper script:
./scripts/run.sh python src/openpredict_model/train.py train-model
```
