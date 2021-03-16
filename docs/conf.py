# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../pylipid'))

# import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'PyLipID'
copyright = '2020, Wanling Song'
author = 'Wanling Song'


# -- General configuration ---------------------------------------------------

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = [
        # Make sure that any autodoc declarations show the right members
        "members",
        "inherited-members",
        "private-members",
        "show-inheritance",
 ]

# autosummary
autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

extensions = [
        'sphinx.ext.intersphinx',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.napoleon',
        'sphinx.ext.coverage',
        'sphinx.ext.mathjax',
        'sphinx.ext.viewcode',
        'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_static_path = ['static']
html_logo = 'static/pylipid_logo.png'
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

exclude_patterns = ['_build', '*_test*', '**/.ipynb_checkpoints/*']

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------




