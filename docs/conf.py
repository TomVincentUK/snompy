# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "finite-dipole"
copyright = "2023, Tom Vincent"
author = "Tom Vincent"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "numpydoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "placeholderlogo.svg",
        "image_dark": "placeholderlogo.svg",
    },
    "github_url": "https://github.com/TomVincentUK/finite-dipole",
    "collapse_navigation": True,
    "navbar_end": ["navbar-icon-links"],
}

html_static_path = ["_static"]
html_css_files = ["finite-dipole.css"]
html_context = {"default_mode": "light"}
