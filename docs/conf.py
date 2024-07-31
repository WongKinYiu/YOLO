# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "YOLO-docs"
copyright = "2024, Kin-Yiu, Wong and Hao-Tang, Tsui"
author = "Kin-Yiu, Wong and Hao-Tang, Tsui"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "linuxdoc.rstFlatTable",
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "sticky_navigation": False,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
