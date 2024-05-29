.. _intro:

Introduction
============

The main purpose of ``pysnom`` is to provide functions to calculate the effective polarizability, :math:`\alpha_{eff}`, of a SNOM tip and a sample, which can be used to predict contrast in SNOM measurements.
It also contains other useful features for SNOM modelling, such as an implementation of the transfer matrix method for calculating far-field reflection coefficients of multilayer samples, and code for simulating lock-in amplifier demodulation of arbitrary functions.

Below on this page are some example scripts, showing idiomatic usage of ``pysnom``.
If you already know how the finite dipole model (FDM) or point dipole model (PDM) works, these might be enough to get started with this package.
You can also refer to the detailed explanations of the functions used in the :doc:`../API/index`.
The rest of this guide will take you through the workings of the two models, and also give tips on how the models can be used to help analyse SNOM data.

Installation
------------

Installation via ``pip``::

   pip install pysnom


Useful third-party packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The examples in this guide rely heavily on ``numpy``, a Python package for eficient numerical computation, which should be installed automatically when you install ``pysnom``.
To follow along, it might also be helpful to install ``matplotlib``, a Python package for data visualisation, and ``scipy``, a Python package for scientific computation.
These can be installed like::

   pip install matplotlib scipy

or for ``conda`` users::

   conda install -c conda-forge matplotlib scipy

Usage examples
--------------

The examples on this page are intended to give a taste of what ``pysnom`` can do, as well as to model idiomatic use of the package.
We've deliberately left out detailed explanations from this section, so don't worry if you don't understand what's going on here yet!
The following pages of this guide should take you through the concepts needed to understand these scripts.

Approach curve on silicon
^^^^^^^^^^^^^^^^^^^^^^^^^

This example uses both the FDM and PDM  to calculate the decay of the SNOM amplitude, :math:`s_n \propto \alpha_{eff, n}`, for different demodulation harmonics, :math:`n` as the SNOM tip is moved in the :math:`z` direction, away from a sample of bulk silicon.

.. plot:: guide/intro/approach.py
   :align: center

Thickness-dependent PMMA spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This more involved example uses the a multilayer FDM to simulate a SNOM spectrum from a thin layer of `poly(methyl methacrylate) <https://en.wikipedia.org/wiki/Poly(methyl_methacrylate)>`_ (PMMA) on silicon, for different thicknesses of PMMA, and normalises the signal to a reference spectrum taken from bulk gold.
It also includes the effects of the far-field reflection coefficient of the sample on the observed SNOM spectra.

.. plot:: guide/intro/spectra.py
   :align: center

(The dielectric function of PMMA in the above example was based on reference [1]_, and the dielectric function of gold was taken from reference [2]_).


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