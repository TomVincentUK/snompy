"""
General tools used by other modules.

References
----------
.. [1] B. Hauer, A.P. Engelhardt, T. Taubner,
   Quasi-analytical model for scattering infrared near-field microscopy on
   layered systems,
   Opt. Express. 20 (2012) 13173.
   https://doi.org/10.1364/OE.20.013173.
"""
import numpy as np
from numba import njit
from scipy.integrate import quad


def complex_quad(func, a, b, **kwargs):
    """
    Wrapper to `scipy.integrate.quad` to allow complex integrands.
    """
    real_part = quad(lambda t, *args: np.real(func(t, *args)), a, b, **kwargs)
    imag_part = quad(lambda t, *args: np.imag(func(t, *args)), a, b, **kwargs)
    return real_part[0] + 1j * imag_part[0], real_part[1] + 1j * imag_part[1]


@njit
def refl_coeff(eps_i, eps_j=1 + 0j):
    """
    Electrostatic reflection coefficient for an interface between materials
    i and j. Defined as :math:`\beta_{ij}`` in equation (7) of reference
    [1]_.

    Parameters
    ----------
    eps_i : complex
        Dielectric function of material i.
    eps_j : complex, default 1 + 0j
        Dielectric function of material j.

    Returns
    -------
    beta_ij : complex
        Electrostatic reflection coefficient of the sample.
    """
    return (eps_i - eps_j) / (eps_i + eps_j) + 0j  # + 0j ensures complex


@njit
def Fourier_envelope(t, n):
    """
    A complex sinusoid with frequency 2 * pi * `n`, to be used in an
    integral that extracts the nth Fourier coefficient.

    Parameters
    ----------
    t : float
        Domain of the function.
    n : int
        Order of the Fourier component.

    Returns
    -------
    sinusoids : complex
        A complex sinusoid with frequency 2 * pi * `n`.
    """
    return np.exp(-1j * n * t)
