import numpy as np

defaults = dict(
    eps_environment=1 + 0j,
    x_0=1.31 * 15 / 17,  # But calculate from `radius` and `semi_maj_axis` when possible
    x_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    N_demod_trapz=64,
    Laguerre_order=64,
)
