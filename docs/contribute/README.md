Documentation to run the **Translator OpenPredict API 🔮🐍**  in development and contribute to the project.

# Instructions for development 📝

Contributions, critics and pull requests are welcomed from anybody 🌍! Follow those instructions if you want to make changes to the OpenPredict source code:

### Clone

```bash
git clone https://github.com/MaastrichtU-IDS/openpredict.git
cd openpredict
```

### Install

This will install `openpredict` and update the package automatically when the files changes locally 🔃

```bash
pip install -e .
```

### Start API for development

Run in development with [Flask 🧪](https://flask.palletsprojects.com/en/1.1.x/). The API will reload automatically at each change 🔃

```bash
openpredict start-api --debug
```

### Test

Run the OpenPredict API tests locally:

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

# Code of Conduct 🤼

Absolutely everyone is welcome to join our project! 🌍 🌏 🌎 🌜

Except from people with harmful intentions towards other human beings, or algorithms.

Because **we want you to read this**, we are going to keep it simple and clean:

* Be nice, please.
* Don't make jokes related to the ethnicity, origins, or community a user belongs to or identifies as, unless it is about windows users, or emacs/vim aliens.
* Critics are encouraged, but please be considerate of other people work and difference of background.
* Accept and answer relevant critics and questions about your approach, it shows interest in your ideas 🤝
* Be careful with confidential informations, including your and others personal informations 🔒
* Respect intellectual property, as much as possible 🧻

> If you are feeling threatened or harassed within this project, **please speak up and contact our team at [ids-contact-l@maastrichtuniversity.nl](mailto:ids-contact-l@maastrichtuniversity.nl)** 🗣️

---

# See also 👀

* The [documentation main page](https://maastrichtu-ids.github.io/translator-openpredict) at **[maastrichtu-ids.github.io/translator-openpredict 🔮🐍](https://maastrichtu-ids.github.io/translator-openpredict)** 
* Browse the [automatically generated Python documentation](https://maastrichtu-ids.github.io/translator-openpredict/docs/package) at **[docs 📖](https://maastrichtu-ids.github.io/translator-openpredict/docs)** 