import numpy as np

defaults = dict(
    eps_env=1 + 0j,
    d_Q0=1.31 * 15 / 17,  # But calculate from `r_tip` and `L_tip` when possible
    d_Q1=0.5,
    r_tip=20e-9,
    L_tip=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    n_trapz=64,
    n_lag=64,
    n_tayl=16,
    beta_threshold=1.01,
)
