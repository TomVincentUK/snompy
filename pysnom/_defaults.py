import numpy as np

defaults = dict(
    eps_environment=1 + 0j,
    d_Q0=1.31 * 15 / 17,  # But calculate from `r_tip` and `L_tip` when possible
    d_Q1=0.5,
    r_tip=20e-9,
    L_tip=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    N_demod_trapz=64,
    laguerre_order=64,
    taylor_order=16,
    beta_threshold=1.01,
)
