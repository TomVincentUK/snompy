"""
Default values (:mod:`pysnom.defaults`)
=======================================

.. currentmodule:: pysnom.defaults

This module holds default values used across various places in ``pysnom``.
"""
import numpy as np

eps_env = 1 + 0j
d_Q0 = 1.31 * 15 / 17
d_Q1 = 0.5
r_tip = 20e-9
L_tip = 300e-9
g_factor = 0.7 * np.exp(0.06j)
n_trapz = 64
n_lag = 64
n_tayl = 16
beta_threshold = 1.01
