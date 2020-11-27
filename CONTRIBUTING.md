# Contributing

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/MaastrichtU-IDS/translator-openpredict/issues) if applicable.

If you are part of the [MaastrichtU-IDS organization on GitHub](https://github.com/MaastrichtU-IDS) you can directly create a branch in this repository. Otherwise you will need to first [fork this repository](https://github.com/MaastrichtU-IDS/translator-openpredict/fork).

To contribute:

1. Clone the repository 📥

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

2. Create a new branch from the `master` branch and add your changes to this branch 🕊️

```bash
git checkout -b my-branch
```

## Development process

Install `openpredict` from the source code, and update the package automatically when the files changes locally :arrows_counterclockwise:

```bash
pip3 install -e .
```

> See the [main README](https://github.com/MaastrichtU-IDS/translator-openpredict) for more details on the package installation.

### Start the OpenPredict API :rocket:

Start the Virtuoso database locally on http://localhost:8890 using Docker (login: `dba` / `dba`):

```bash
docker-compose up -d --force-recreate
```

Start the **OpenPredict API in debug mode** on http://localhost:8808 (the API will be reloaded automatically at each change to the code)

```bash
openpredict start-api --debug
```

> Stop the Virtuoso container:
>
> ```bash
> docker-compose down
> ```

### Reset your local OpenPredict data

Use the `reset_openpredict.sh` script to delete the folders where the OpenPredict API and Virtuoso data are stored, in `data/virtuoso` and `data/openpredict`

```bash
./reset_openpredict.sh
```

> This command uses `sudo` to be able to delete the `data/virtuoso` folder which has been created by the `docker` user.
>
> On Windows: delete all files in `data` folder, just keep `initial-openpredict-metadata.ttl` 

> See more **[documentation to deploy the OpenPredict API](https://github.com/MaastrichtU-IDS/translator-openpredict/tree/master/docs)** locally or with Docker.

## Pull Request process

1. Ensure the test are passing before sending a pull request 🧪
```
pip3 install pytest
pytest tests
```
2. Update the `README.md` with details of changes, this includes new environment variables, exposed ports, useful file locations and container parameters 📝
3. [Send a pull request](https://github.com/MaastrichtU-IDS/translator-openpredict/compare) to the `master` branch, answer the questions in the pull request message 📤
4. Project contributors will review your change as soon as they can ✔️

## Versioning process

The versioning scheme for new releases on GitHub used is [SemVer](http://semver.org/) (Semantic Versioning).

Change version in `setup.py` before new release.
