# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'magnet'
copyright = '2023, Prismadic'
author = 'Prismadic'
release = '0.0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo'
    , 'sphinx.ext.viewcode'
    , 'sphinx.ext.autodoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

github_url = "https://github.com/Prismadic/magnet"
display_github = True
html_logo = "../magnet.png"
pygments_style = 'default'
version = "v0.0.9"
release = "latest"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 5,
}
html_show_sourcelink = True
html_static_path = ['_static']
