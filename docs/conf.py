import math

from cycler import cycler

project = "snompy"
copyright = "2024, Tom Vincent"
author = "Tom Vincent"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_togglebutton",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/snompy_logo.svg"
html_favicon = "_static/snompy_favicon.svg"

html_theme_options = {
    "github_url": "https://github.com/TomVincentUK/snompy",
    "secondary_sidebar_items": ["page-toc"],
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "logo": {
        "image_light": "snompy_logo.svg",
        "image_dark": "snompy_logo.svg",
    },  # Not sure why this is needed, but it errors without
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_context = {"default_mode": "light"}

autosummary_generate = True
autosummary_imported_members = True
numpydoc_class_members_toctree = False

add_function_parentheses = True
numpydoc_use_plots = True

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
            "#9e0142",
            "#3288bd",
            "#f46d43",
            "#66c2a5",
            "#d53e4f",
            "#fdae61",
            "#5e4fa2",
        ]
    ),
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 144,
    "figure.figsize": (3.5 * phi, 3.5),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "figure.facecolor": (0, 0, 0, 0),
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
}
plot_apply_rcparams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

plot_html_show_formats = False
plot_html_show_source_link = False
