"""
Finite dipole model
===================

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the finite dipole
model (FDM).

"""
from . import bulk, combo, multi

__all__ = ["bulk", "multi", "combo"]
