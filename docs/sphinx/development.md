# Documentation development

This project ships two documentation pathways:

- **Sphinx** for rendered HTML documentation built from reStructuredText,
  Markdown, and docstrings.
- **pydoc** for lightweight, module-level HTML output.

## Build with Sphinx

From the repository root:

```bash
pip install -r requirements-dev.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

Open `docs/sphinx/_build/html/index.html` in a browser to view the output.

## Generate pydoc HTML

From the repository root:

```bash
python -m pydoc -w tsp.tsp
```

This writes `tsp.tsp.html` in the working directory. Move it into a shared
location if you want to publish it alongside the Sphinx output.
