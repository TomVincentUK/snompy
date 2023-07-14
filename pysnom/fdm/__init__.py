"""
Finite dipole model (:mod:`pysnom.fdm`)
=======================================

.. currentmodule:: pysnom.fdm

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the finite dipole
model (FDM).

Bulk finite dipole model
------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bulk.eff_pol_n
    bulk.eff_pol
    bulk.geom_func

Taylor series representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bulk.refl_coef_qs
    bulk.eff_pol_n_taylor
    bulk.taylor_coef
    bulk.geom_func_taylor

Multilayer finite dipole model
------------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    multi.eff_pol_n
    multi.eff_pol
    multi.geom_func
    multi.eff_pos_and_charge
    multi.phi_E_0

"""
from . import bulk, multi

__all__ = ["bulk", "multi"]
