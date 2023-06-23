import numpy as np

defaults = dict(
    eps_environment=1 + 0j,
    d_Q0=1.31
    * 15
    / 17,  # But calculate from `radius` and `semi_maj_axis` when possible
    d_Q1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    N_demod_trapz=64,
    laguerre_order=64,
    taylor_order=16,
    beta_threshold=1.01,
)
