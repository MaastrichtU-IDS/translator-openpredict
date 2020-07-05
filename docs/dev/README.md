Documentation to run the **Translator OpenPredict API** in development.

Contributions, feedbacks and pull requests are welcomed from anyone!

# Install for dev 📥

Follow those instructions if you want to run **OpenPredict** in development, and more easily make changes to the source code:

### Clone

```bash
git clone https://github.com/MaastrichtU-IDS/translator-openpredict.git
cd translator-openpredict
```

### Install

This will install `openpredict` and update the package automatically when the files changes locally 🔃

```bash
pip install -e .
```

### Enable autocomplete

Enabling command line autocomplete in the terminal provides a better experience using the CLI ⌨️ 

* If you use `ZSH`: add a line to the `~/.zshrc` file.

```bash
echo 'eval "$(_OPENPREDICT_COMPLETE=source_zsh openpredict)"' >> ~/.zshrc
```

* If you use `Bash`: add a line to the `~/.bashrc` file. 

  ```bash
  echo 'eval "$(_OPENPREDICT_COMPLETE=source openpredict)"' >> ~/.bashrc
  ```

  > Bash autocomplete needs to be tested.

# Run in dev 🚧

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

### Generate documentation

Documentation in [docs/ 📖](docs/)  generated using [pydoc-markdown](https://pydoc-markdown.readthedocs.io/en/latest/)

```bash
pip install pydoc-markdown
```

Generate markdown documentation page for the `openpredict` package in `docs/`

```bash
pydoc-markdown --render-toc -p openpredict > docs/README.md
```

> This can also be done using Sphinx, see this article on [deploying Sphinx to GitHub Pages](https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/)
>
> ```bash
> pip install sphinx
> sphinx-quickstart docs/ --project 'openpredict' --author 'Vincent Emonet'
> cd docs/
> make html
> ```

---

# See also 👀

* **[Documentation main page 🔮🐍](https://maastrichtu-ids.github.io/translator-openpredict)**
* **[Documentation generated from the source code 📖](https://maastrichtu-ids.github.io/translator-openpredict/docs)**
* **[Code of Conduct 🤼](https://github.com/MaastrichtU-IDS/translator-openpredict/blob/master/CODE_OF_CONDUCT.md)**