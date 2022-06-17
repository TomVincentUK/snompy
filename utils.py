import numpy as np
from scipy.integrate import quad
from tqdm import tqdm


def complex_quad(func, a, b, **kwargs):
    """
    Wrapper to `scipy.integrate.quad` to allow complex integrands.
    """
    real_part = quad(lambda t, *args: np.real(func(t, *args)), a, b, **kwargs)
    imag_part = quad(lambda t, *args: np.imag(func(t, *args)), a, b, **kwargs)
    return real_part[0] + 1j * imag_part[0], real_part[1] + 1j * imag_part[1]


def tqdm_nd(shape, **kwargs):
    """
    Creates an n-dimensional tqdm iterator. Iterator yields tuple of indices.
    kwargs passed to tqdm.
    """
    return tqdm(list(np.ndindex(shape)), **kwargs)
