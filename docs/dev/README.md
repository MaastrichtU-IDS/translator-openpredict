Documentation to run the **Translator OpenPredict API** in development.

> Contributions, [feedbacks](https://github.com/MaastrichtU-IDS/translator-openpredict/issues) and pull requests are welcomed!

This repository uses [GitHub Actions](https://github.com/MaastrichtU-IDS/translator-openpredict/actions) to:

* Automatically run tests at each push to the `master` branch
* Publish the [OpenPredict package to PyPI](https://pypi.org/project/openpredict/) when a release is created (N.B.: the version of the package needs to be increased in [setup.py](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/setup.py#L6) before).

# Alternative: install for dev 📥

Install `openpredict` locally, if you want to run **OpenPredict** in development, make changes to the source code, and build new models.

### Clone

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

### Install

This will install `openpredict` and update the package automatically when the files changes locally 🔃

```bash
pip3 install -e .
```

# Run in dev 🚧

### Define environment variable

For triplestore credentials and the APIKEY to use when querying the API

```bash
export OPENPREDICT_USERNAME=import_user
export OPENPREDICT_PASSWORD=password
export OPENPREDICT_APIKEY=myapikey
```

### Start API

Use the CLI to run in development with [Flask 🧪](https://flask.palletsprojects.com/en/1.1.x/). The API will reload automatically at each change 🔃

```bash
openpredict start-api --debug
```

### Run tests

Run the **OpenPredict API** tests locally:

```bash
pytest tests
```

Run a specific test in a file, and display `print` in the output:

```bash
pytest tests/test_openpredict_api.py::test_post_reasoner_predict -s
```

# Create a new API service

1. Create the operations in the [openpredict/openapi.yml](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/openpredict/openapi.yml#L44) file

Provide the path to the function that will resolve this API call:

```yaml
paths:
  /predict:
    get:
      operationId: openpredict.openpredict_api.get_predict
      parameters:
      - name: entity
        in: query
        description: CURIE of the entity to process (e.g. drug, disease, etc)
        example: DRUGBANK:DB00394
        required: true
        schema:
          type: string
```

2. Now, create the function in the [openpredict/openpredict_api.py](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/openpredict/openpredict_api.py#L67) file

```python
def get_predict(entity, classifier="Predict OMIM-DrugBank", score=None, n_results=None):
    print("Do stuff with " + classifier)
```

> The parameters provided in openapi.yml and the one of the function in openpredict_api.py need to match!

# Generate documentation 📖

Documentation in [docs/](docs/)  generated from the Python source code docstrings using [pydoc-markdown](https://pydoc-markdown.readthedocs.io/en/latest/).

```bash
pip3 install pydoc-markdown
```

Generate markdown documentation page for the `openpredict` package in `docs/`

```bash
pydoc-markdown --render-toc -p openpredict > docs/README.md
```

Modify the generated page title:

```bash
find docs/README.md -type f -exec sed -i "s/# Table of Contents/# OpenPredict Package documentation 🔮🐍/g" {} +
```

> This can also be done using Sphinx, see this article on [deploying Sphinx to GitHub Pages](https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/)
>
> ```bash
> pip3 install sphinx
> sphinx-quickstart sphinx-docs/ --project 'openpredict' --author 'Vincent Emonet'
> cd sphinx-docs/
> make html
> ```

## More about the data model

The data model to represent runs, models evaluation, features, etc is based on the Machine Learning Schema. See the [ML Schema documentation](http://ml-schema.github.io/documentation/ML%20Schema.html)

> See the OpenPredict diagram in `docs/OpenPREDICT_datamodel.jpg`

# See also 👀

* **[Documentation main page 🔮🐍](https://maastrichtu-ids.github.io/translator-openpredict)**
* **[Documentation generated from the source code 📖](https://maastrichtu-ids.github.io/translator-openpredict/docs)**

