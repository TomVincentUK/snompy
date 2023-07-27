.. _sample:

Working with samples
====================

* Specifying samples in pysnom

  * Samples are layered and extend infinitely
  * Sample object can be specified by eps_stack, t_stack
  * Sample object returns quasistatic reflection coefficients
  * Sample object can be specified by beta_stack
  * eps or beta, and t can be different shapes: produces expected shape for output
  * Sample can calculate far-field Fresnel reflection for different angles, but needs to be told k_vac
  * Can also give k_vac as an argument at initialisation, as it doesn't usually change
  * Bulk samples can be made easily using :func:`pysnom.sample.bulk_sample`.

.. doctest::

   >>> sample = pysnom.sample.bulk_sample([10, 20])
   >>> print(sample.beta_stack)
   [0.81818182+0.j]