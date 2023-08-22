import numpy as np


def _pad_for_broadcasting(array, broadcast_with):
    """Pads `array` with singleton dimensions so that it broadcasts with
    all arrays in `broadcast_with` along first axis.
    """
    index_pad_dims = np.max([np.ndim(a) for a in broadcast_with])
    return np.asanyarray(array).reshape(-1, *(1,) * index_pad_dims)
