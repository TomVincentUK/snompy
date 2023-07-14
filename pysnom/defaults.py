"""
Default values (:mod:`pysnom.defaults`)
=======================================

.. currentmodule:: pysnom.defaults

This module holds default values used by various functions in ``pysnom``.
"""
import numpy as np

# Sample-related properties
eps_env = 1 + 0j

# Tip-related properties
r_tip = 20e-9
L_tip = 300e-9
g_factor = 0.7 * np.exp(0.06j)
d_Q1 = 0.5

# Demodulation-related properties
n_trapz = 64

# q integral-related properties
n_lag = 64

# Taylor inversion-related properties
n_tayl = 16
beta_threshold = 1.01
