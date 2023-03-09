:html_theme.sidebar_secondary.remove:

.. _user_guide:

User guide
==========

This is the user guide for the Python package ``pysnom``, which is
an implementation of the finite dipole model (FDM) and the point dipole
model (PDM) for predicting contrasts in scattering-type scanning near-field
optical microscopy (SNOM) measurements.

Here we provide examples of how to use this package, along with an
explanation of how the models are implemented.
You won't need a detailed understanding of the FDM or PDM to understand
this guide, but it'll be helpful to know the basics of SNOM and atomic
force microscopy (AFM).
For more detailed documentation of the functions provided by this package,
see :ref:`API`.

.. toctree::
   :maxdepth: 1
   :hidden:

   intro
   scattering
   bulk_fdm
   multi_fdm
   bulk_pdm
   demodulation

Foundations
-----------

These pages apply equally well to both FDM and PDM.
Start here for a general introduction to modelling with ``pysnom``.

.. grid::

    .. grid-item-card::
      :link: intro
      :link-type: ref

      Introduction
      ^^^^^^^^^^^^

      The best place to get started.
      Here we cover how to install the package, and show some examples to
      give a taste of what ``pysnom`` can do.

    .. grid-item-card::
      :link: scattering
      :link-type: ref

      Modelling SNOM scattering
      ^^^^^^^^^^^^^^^^^^^^^^^^^

      Here we introduce the concept of effective polarisability, and how
      it's used to model SNOM scattering.

    .. grid-item-card::
      :link: demodulation
      :link-type: ref

      Demodulation
      ^^^^^^^^^^^^

      Here you can find a detailed  guide to demodulation: why it's needed,
      and how it's implemented in ``pysnom``

The models
----------

These pages explain the different models used by ``pysnom``, and how
they're implemented.

.. grid::

    .. grid-item-card::
      :link: bulk_fdm
      :link-type: ref

      The finite dipole model (FDM)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      The standard method for modelling SNOM, for bulk samples.
      This model has a good quantitative agreement with experimental
      results.

    .. grid-item-card::
      :link: multi_fdm
      :link-type: ref

      The multilayer FDM
      ^^^^^^^^^^^^^^^^^^

      An extension of the FDM, which copes with samples with multiple
      layers.
      Start with the bulk FDM before reading this page.

    .. grid-item-card::
      :link: bulk_pdm
      :link-type: ref

      The point dipole model (PDM)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      An older method of modelling SNOM.
      It's simpler, but generally less quantitative than the FDM.

Advanced topics
---------------

These pages don't exist yet!
But when they do, they'll probably include things like fitting to the FDM.