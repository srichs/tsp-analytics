import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "python-tsp-priv"
author = "python-tsp-priv contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "alabaster"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
