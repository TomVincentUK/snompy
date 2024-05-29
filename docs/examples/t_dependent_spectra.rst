Thickness-dependent PMMA spectra
================================

This example uses the a multilayer FDM to simulate a SNOM spectrum from a thin layer of `poly(methyl methacrylate) <https://en.wikipedia.org/wiki/Poly(methyl_methacrylate)>`_ (PMMA) on silicon, for different thicknesses of PMMA, and normalises the signal to a reference spectrum taken from bulk gold.
It also includes the effects of the far-field reflection coefficient of the sample on the observed SNOM spectra.

.. plot:: examples/scripts/t_dependent_spectra.py
   :align: center

(The dielectric function of PMMA in the above example was based roughly on reference [1]_, and the dielectric function of gold was taken from reference [2]_).


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