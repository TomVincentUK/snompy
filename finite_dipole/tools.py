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
from scipy.integrate import trapezoid, simpson
from numba import njit


@njit
def refl_coeff(eps_i, eps_j):
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
    eps_i = np.asarray(eps_i)
    eps_j = np.asarray(eps_j)
    return (eps_j - eps_i) / (eps_j + eps_i)


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


@njit
def _sampled_integrand(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    x = x_0 + x_amplitude * np.cos(theta)
    f = f_x(x, *f_args)
    envelope = np.exp(-1j * harmonic * theta)
    return f * envelope


def demodulate(
    f_x,
    x_0,
    x_amplitude,
    harmonic,
    f_args=(),
    method="trapezium",
    n_samples=64,
):
    if method not in ["trapezium", "simpson", "adaptive"]:
        raise ValueError("`method` must be 'trapezium', 'simpson' or 'adaptive'.")

    x_0, x_amplitude, harmonic, *f_args = np.broadcast_arrays(
        *(x_0, x_amplitude, harmonic) + f_args
    )
    f_args = tuple(f_args)

    if method == "adaptive":
        raise NotImplementedError()
    else:
        x_0, x_amplitude, harmonic, *f_args = [
            np.array(arr)[..., np.newaxis]
            for arr in np.broadcast_arrays(*(x_0, x_amplitude, harmonic) + f_args)
        ]
        f_args = tuple(f_args)

        integrand = _sampled_integrand(
            f_x, x_0, x_amplitude, harmonic, f_args, n_samples
        )

        if method == "trapezium":
            result = trapezoid(integrand) * 2 * np.pi / (n_samples - 1)
        elif method == "simpson":
            result = simpson(integrand) * 2 * np.pi / (n_samples - 1)

    return result
