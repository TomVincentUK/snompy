Introduction
============

The main function of this package is to provide functions to calculate the
effective polarisability, :math:`\alpha_{eff}`, of an AFM tip and a sample,
which can be used to predict contrast in SNOM measurements.
It also provides a function to simulate lock-in amplifier measurements of
signals at multiple harmonics, which is an important part of modelling a
SNOM signal.

Below on this page are some example scripts, showing idiomatic usage of
``pysnom``.
If you already know how the FDM works, these might be enough to get started
with this package.
You can also refer to the detailed explanations of the functions used in
the API.
The rest of this guide will take you through the workings of FDM, and also
give tips on how the model can be used to help analyse SNOM data.

The examples in this guide rely heavily on ``numpy``, a Python package for
eficient numerical computation, which should be installed automatically
when you install ``pysnom``.
To follow along, it might also be helpful to install ``matplotlib``, a
Python package for data visualisation, and ``scipy``, a Python package for
scientific computation.

.. LINKS TO PACKAGES ?

Installation
------------

Currently, this project is still in development, and not yet public. So to
install ``pysnom`` you'll need to clone the git repository to your
local environment then install in development mode like:

.. code-block:: bash

   git clone https://github.com/TomVincentUK/pysnom.git
   cd pysnom
   pip install -e .

When the project goes public, I'll upload it to PyPI. When I do, the
installation should look like:

.. code-block:: bash

   pip install pysnom

Eventually, I'll also look into adding it to ``conda-forge``, so it can be
installed like:

.. code-block:: bash

   conda install -c conda-forge pysnom


Usage examples
--------------

These examples are intended to give a taste of what ``pysnom`` can
do, as well as to model idiomatic use of the package.
We've deliberately left out detailed descriptions from this section, as
these will be given in the following pages of this guide.

Approach curve on silicon
^^^^^^^^^^^^^^^^^^^^^^^^^

The example below calculates the decay of the SNOM amplitude,
:math:`s_n \propto \alpha_{eff, n}`, for different demodulation harmonics,
:math:`n` as the sample is moved in the :math:`z` direction, away from a
sample of bulk silicon.

.. plot:: guide/plots/intro_approach.py
   :align: center

This shows the expected result: that higher order demodulation leads to a
faster decay of the SNOM signal (*i.e.* stronger surface confinement).
It also shows the quantitative difference between approach curves
calculated with the point dipole model (PDM) and finite dipole model (FDM).

Thickness-dependent PMMA spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example simulates a SNOM spectrum from a multilayer structure of
poly(methyl methacrylate) (PMMA) on silicon, for different thicknesses of
PMMA, and normalises the signal to a reference spectrum taken from gold.

.. plot:: guide/plots/intro_spectra.py
   :align: center

(The dielectric function of PMMA in the above example was based on
reference [1]_, and the dielectric function of gold was taken from
reference [2]_).


References
----------

.. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and R.
   Hillenbrand, “Subsurface chemical nanoidentification by nano-FTIR
   spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359, Dec. 2020,
   doi: 10.1038/s41467-020-17034-6.
.. [2] M. A. Ordal et al., “Optical properties of the metals Al, Co, Cu,
   Au, Fe, Pb, Ni, Pd, Pt, Ag, Ti, and W in the infrared and far infrared,”
   Appl. Opt., vol. 22, no. 7, p. 1099, Apr. 1983,
   doi: 10.1364/AO.22.001099.