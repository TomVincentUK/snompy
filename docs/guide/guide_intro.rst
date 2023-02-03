User guide
==========

WRITE ME.

The structure of this guide might look something like this:

.. toctree::
   :maxdepth: 3

   Bulk finite dipole model <bulk>
   Multilayer finite dipole model <multilayer>

Usage examples
--------------

This package can be used for simple simulations such as in the following
example, which calculates the decay of the SNOM amplitude,
:math:`s_n \propto \alpha_{eff, n}`, for different demodulation harmonics,
:math:`n` as the sample is moved in the :math:`z` direction, away from a
sample of bulk silicon.

.. plot:: guide/plots/intro_approach.py
   :align: center

This shows the expected result: that higher order demodulation leads to a
faster decay of the SNOM signal (*i.e.* stronger surface confinement).

The package can also be used for more involved simulations, such as the
script below, which simulates a SNOM spectrum from a multilayer structure
of poly(methyl methacrylate) (PMMA) on silicon, for different thicknesses
of PMMA, and normalises the signal to a reference spectrum taken from gold.

.. plot:: guide/plots/intro_spectra.py
   :align: center

Installation
------------

I've not yet uploaded to PyPi. But when I do, the installation should look
like:

.. code-block:: bash

   pip install finite-dipole

Eventually, I'll also look into adding it to ``conda-forge``, so it can be
installed like:

.. code-block:: bash

   conda install -c conda-forge finite-dipole

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