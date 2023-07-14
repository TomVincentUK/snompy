import numpy as np

from . import defaults


def _pad_for_broadcasting(array, broadcast_with):
    """Pads `array` with singleton dimensions so that it broadcasts with
    all arrays in `broadcast_with` along first axis.
    """
    index_pad_dims = np.max([np.ndim(a) for a in broadcast_with])
    return np.asarray(array).reshape(-1, *(1,) * index_pad_dims)


def _fdm_defaults(r_tip, L_tip, g_factor, d_Q0, d_Q1):
    r_tip = defaults.r_tip if r_tip is None else r_tip
    L_tip = defaults.L_tip if L_tip is None else L_tip
    g_factor = defaults.g_factor if g_factor is None else g_factor
    if d_Q0 is None:
        d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)
    d_Q1 = defaults.d_Q1 if d_Q1 is None else d_Q1
    return r_tip, L_tip, g_factor, d_Q0, d_Q1
