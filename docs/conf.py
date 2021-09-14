# Sphinx configuration file
#
# For a full list of available options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project   = 'Nyström kernel PCA'
copyright = '2021, Fredrik Hallgren'
author    = 'Fredrik Hallgren'
release   = '1.0.2'
version   = '1.0.2'


# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc', 'numpydoc', 'sphinx_rtd_theme']

templates_path = ['_templates']

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"

autodoc_typehints = 'none'

numpydoc_show_class_members = False

add_module_names = False

