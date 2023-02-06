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
    'sphinx.ext.intersphinx',
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
    "font.size": 12,
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
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

plot_html_show_formats = False
plot_html_show_source_link = False
numpydoc_use_plots = True
