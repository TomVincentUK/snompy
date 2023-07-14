"""
Default values
==============

This module provides an object containing default values used by various
functions in ``pysnom``.
"""
import numpy as np


class Defaults:
    """A class holding default values used by various functions in
    ``pysnom``.
    """

    def __init__(
        self,
        eps_env=1 + 0j,
        r_tip=20e-9,
        L_tip=300e-9,
        g_factor=0.7 * np.exp(0.06j),
        d_Q1=0.5,
        n_trapz=64,
        n_lag=64,
        n_tayl=16,
        beta_threshold=1.01,
    ):
        # Sample-related properties
        self.eps_env = eps_env

        # Tip-related properties
        self.r_tip = r_tip
        self.L_tip = L_tip
        self.g_factor = g_factor
        self.d_Q1 = d_Q1

        # Demodulation-related properties
        self.n_trapz = n_trapz

        # q integral-related properties
        self.n_lag = n_lag

        # Taylor inversion-related properties
        self.n_tayl = n_tayl
        self.beta_threshold = beta_threshold

    def _fdm_defaults(self, r_tip, L_tip, g_factor, d_Q0, d_Q1):
        r_tip = self.r_tip if r_tip is None else r_tip
        L_tip = self.L_tip if L_tip is None else L_tip
        g_factor = self.g_factor if g_factor is None else g_factor
        if d_Q0 is None:
            d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)
        d_Q1 = self.d_Q1 if d_Q1 is None else d_Q1
        return r_tip, L_tip, g_factor, d_Q0, d_Q1


defaults = Defaults()
