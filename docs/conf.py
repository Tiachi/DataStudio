# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "DataStudio"
copyright = "2024, DataStudio Team"
author = "DataStudio Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",          # Support Google/NumPy docstring style
    "sphinx.ext.viewcode",          # Add links to source code
    "sphinx.ext.inheritance_diagram",  # Class hierarchy diagrams
    "sphinx.ext.intersphinx",       # Link to external docs
    "sphinx_autodoc_typehints",     # Better type hint rendering
    "myst_parser",                  # Markdown support
    "sphinx_copybutton",            # Copy button for code blocks
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,       # Show class inheritance
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Inheritance diagram settings
inheritance_graph_attrs = dict(rankdir="TB", size='"6.0, 8.0"', fontsize=14)
inheritance_node_attrs = dict(shape="box", fontsize=14, color="dodgerblue")

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for inheritance diagrams ----------------------------------------
# Requires graphviz to be installed
graphviz_output_format = "svg"
