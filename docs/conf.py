# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import math

project = "finite-dipole"
copyright = "2023, Tom Vincent"
author = "Tom Vincent"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/TomVincentUK/finite-dipole",
    "secondary_sidebar_items": ["page-toc"],
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright"],
    "logo": {
        "image_light": "placeholderlogo.svg",
        "image_dark": "placeholderlogo.svg",
    },  # Not sure why this is needed, but it errors without
}

html_static_path = ["_static"]
html_logo = "_static/placeholderlogo.svg"
html_favicon = "_static/placeholderlogo.svg"
html_css_files = ["finite-dipole.css"]
html_context = {"default_mode": "light"}

autosummary_generate = True
add_function_parentheses = False

plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [("png", 100), "pdf"]

phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

plot_html_show_formats = False
plot_html_show_source_link = False
numpydoc_use_plots = True
