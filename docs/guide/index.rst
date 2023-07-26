.. _user_guide:

User guide
==========

This is the user guide for the Python package ``pysnom``, which is an implementation of the finite dipole model (FDM) and the point dipole model (PDM) for predicting contrasts in scattering-type scanning near-field optical microscopy (SNOM) measurements.

Here we provide examples of how to use this package, along with an explanation of how the models are implemented.
You won't need a detailed understanding of the FDM or PDM to understand this guide, but it'll be helpful to know the basics of SNOM and atomic force microscopy (AFM).
For more detailed documentation of the functions provided by this package, see :ref:`API`.

NOTE TO SELF: user guide plan
-----------------------------

* Intro:

  * Installation
  * Extra modules
  * Flashy examples

* Basics of SNOM modelling:

  * Sample, tip, incident and scattered light:

    * Dipole formed by the coupled tip and sample scatters near-field into far-field
    * Far-field factor from sample

* Specifying samples in pysnom

  * Samples are layered and extend infinitely
  * Sample object can be specified by eps_stack, t_stack
  * Sample object returns quasistatic reflection coefficients
  * Sample object can be specified by beta_stack
  * eps or beta, and t can be different shapes: produces expected shape for output
  * Sample can calculate far-field Fresnel reflection for different angles, but needs to be told k_vac
  * Can also give k_vac as an argument at initialisation, as it doesn't usually change
  * Bulk samples can be made easily using :func:`pysnom.sample.bulk_sample`.

.. toctree::
   :maxdepth: 1

   intro
   scattering
   bulk_fdm
   multi_fdm
   bulk_pdm
   demodulation
