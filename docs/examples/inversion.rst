Inverting the FDM
=================

This example inverts the finite dipole model (FDM) using ``scipy``'s  `optimize.minimize` and by using a built-in method provided by ``snompy`` that uses a Taylor series representation of the effective polarizability.
The taylor series representation only works for samples with weak light matter interaction.

.. plot:: examples/scripts/inversion.py
   :align: center