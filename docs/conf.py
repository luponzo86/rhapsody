# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'rhapsody'
copyright = '2019, Luca Ponzoni'
author = 'Luca Ponzoni'

# The full version, including alpha/beta/rc tags
release = '0.9'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_backup', 'Thumbs.db', '.DS_Store', '**tar.gz**']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The master toctree document.
master_doc = 'index'

# autodoc
autodoc_default_options = {
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': False,
    'imported-members': False,
}
# autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    # if not specified, readthedocs.org uses their theme by default
    html_theme = "sphinx_rtd_theme"
    html_theme_path = ["_themes", ]
else:
    html_theme = "sphinx_rtd_theme"
    html_theme_path = ["_themes", ]

    # import sphinx_theme
    # html_theme = 'stanford_theme'
    # html_theme_path = [sphinx_theme.get_html_theme_path('stanford_theme')]

    # html_theme = 'bootstrap'
    # html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

    # others: 'neo_rtd_theme', 'alabaster', 'pyramid', 'nature'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    # 'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#b7270b',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
