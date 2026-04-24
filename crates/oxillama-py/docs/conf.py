"""Sphinx configuration for oxillama-py documentation."""

import os
import sys

# Point autodoc at the Python package (stubbed — the .pyi file will be used)
sys.path.insert(0, os.path.abspath("../python"))

project = "oxillama-py"
author = "COOLJAPAN OU (Team Kitasan)"
copyright = "2026, COOLJAPAN OU (Team Kitasan)"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "OxiLLaMa Python Bindings"
