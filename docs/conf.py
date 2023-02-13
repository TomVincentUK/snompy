# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import math

from cycler import cycler

project = "pysnom"
copyright = "2023, COPYRIGHT HOLDERS"
author = "Tom Vincent"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/TomVincentUK/pysnom",
    "secondary_sidebar_items": ["page-toc"],
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright"],
    "logo": {
        "image_light": "pysnom_logo.svg",
        "image_dark": "pysnom_logo.svg",
    },  # Not sure why this is needed, but it errors without
}

html_static_path = ["_static"]
html_logo = "_static/pysnom_logo.svg"
html_favicon = "_static/pysnom_favicon.svg"
html_css_files = ["pysnom.css"]
html_context = {"default_mode": "light"}

autosummary_generate = True
add_function_parentheses = True

plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [("png", 100), "pdf"]

phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    "font.size": 12,
    "axes.prop_cycle": cycler(
        color=[
            "#3288bd",
            "#f46d43",
            "#66c285",
            "#d53ef4",
            "#5e4fa2",
            "#fdae61",
            "#9e0142",
        ]
    ),
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 144,
    "figure.figsize": (3.5 * phi, 3.5),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

plot_html_show_formats = False
plot_html_show_source_link = False
numpydoc_use_plots = True
