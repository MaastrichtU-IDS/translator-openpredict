[![Version](https://img.shields.io/pypi/v/nanopub)](https://pypi.org/project/nanopub) [![Python versions](https://img.shields.io/pypi/pyversions/nanopub)](https://pypi.org/project/nanopub) [![Pull requests welcome](https://img.shields.io/badge/pull%20requests-welcome-brightgreen)](https://github.com/fair-workflows/nanopub/fork)

[![Python application](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml){:target="_blank"} [![Publish](https://github.com/fair-workflows/nanopub/actions/workflows/pypi.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/pypi.yml){:target="_blank"} [![cffconvert](https://github.com/fair-workflows/nanopub/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/cffconvert.yml){:target="_blank"}


## 📥 Install for development

Clone the repository and go in the project folder:

```bash
git clone https://github.com/fair-workflows/nanopub
cd nanopub
```

To install the project for development you can either use [`venv`](https://docs.python.org/3/library/venv.html) to create a virtual environment yourself, or use [`hatch`](https://hatch.pypa.io) to automatically handle virtual environments for you.

=== "venv"

    Create the virtual environment in the project folder :

    ```bash
    python3 -m venv .venv
    ```

    Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

    Install all dependencies required for development:

    ```bash
    pip install -e ".[dev,doc,test]"
    ```

    Install `pre-commit` to enable automated formatting and linting of the code at each commit:

    ```bash
    pre-commit install
    ```

=== "hatch"

    Install [Hatch](https://hatch.pypa.io), this will automatically handle virtual environments and make sure all dependencies are installed when you run a script in the project:

    ```bash
    pip install hatch
    ```

    ??? note "Optionally you can improve `hatch` terminal completion"

        See the [official documentation](https://hatch.pypa.io/latest/cli/about/#tab-completion) for more details. For ZSH you can run these commands:

        ```bash
        _HATCH_COMPLETE=zsh_source hatch > ~/.hatch-complete.zsh
        echo ". ~/.hatch-complete.zsh" >> ~/.zshrc
        ```


## 🧑‍💻 Development workflow

=== "venv"

    Try to sign a nanopublication with the code defined in `scripts/dev.py` to test your changes:

    ```bash
    ./scripts/dev.sh
    ```

    The code will be automatically formatted when you commit your changes using `pre-commit`. But you can also run the script to format the code yourself:

    ```bash
    ./scripts/format.sh
    ```

    Check the code for errors, and if it is in accordance with the PEP8 style guide, by running `flake8` and `mypy`:

    ```bash
    ./scripts/lint.sh
    ```

=== "hatch"

    Try to sign a nanopublication with the code defined in `scripts/dev.py` to test your changes:

    ```bash
    hatch run dev
    ```

    The code will be automatically formatted when you commit your changes using `pre-commit`. But you can also run the script to format the code yourself:

    ```bash
    hatch run format
    ```

    Check the code for errors, and if it is in accordance with the PEP8 style guide, by running `flake8` and `mypy`:

    ```bash
    hatch run lint
    ```


## ✅ Run the tests

[![Python application](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml){:target="_blank"}

Tests are automatically run by a GitHub Actions workflow when new code is pushed to the GitHub repository.

The tests use the [```nanopub-java```](https://github.com/Nanopublication/nanopub-java) tool for validating the signing process implemented in python produces similar nanopublications. This is automatically installed by the library, just make sure `java` is available where you run the tests.

=== "venv"

	Run the tests locally:

	```bash
	./scripts/test.sh
	```

	You can also run only a specific test:

	```bash
	./scripts/test.sh tests/test_nanopub.py::test_nanopub_sign_uri
	```

=== "hatch"

	Run the tests locally:

	```bash
	hatch run test
	```

	You can also run only a specific test:

	```bash
	hatch run test tests/test_nanopub.py::test_nanopub_sign_uri
	```


## 📖 Generate docs

[![Publish docs](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/build.yml){:target="_blank"}

The documentation (this website) is automatically generated from the markdown files in the `docs` folder and python docstring comments, and published by a GitHub Actions workflow.

Serve the docs on [http://localhost:8008](http://localhost:8008){:target="_blank"}

=== "venv"

    ```bash
    ./scripts/docs.sh
    ```

=== "hatch"

    ```bash
    hatch run docs
    ```


## 🏷️ Publish a new release

[![Publish to PyPI](https://github.com/fair-workflows/nanopub/actions/workflows/pypi.yml/badge.svg)](https://github.com/fair-workflows/nanopub/actions/workflows/pypi.yml){:target="_blank"}

1. Increment the `__version__` in `nanopub/_version.py`
2. Push to GitHub
3. Create a new release on GitHub
4. A GitHub Action workflow will automatically publish the new version to PyPI
