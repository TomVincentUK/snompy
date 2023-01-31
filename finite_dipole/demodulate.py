"""
Demodulation (:mod:`finite_dipole.demodulate`)
==============================================

.. currentmodule:: finite_dipole.demodulate

This module provides a function to simulate lock-in amplifier measurements
of arbitrary functions.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    demod
"""

import numpy as np
from scipy.integrate import quad_vec, simpson, trapezoid


def _sampled_integrand(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    max_dim = np.max([np.ndim(arg) for arg in (x_0, x_amplitude, harmonic, *f_args)])
    theta = np.linspace(-np.pi, np.pi, n_samples).reshape(
        -1, *np.ones(max_dim, dtype=int)
    )
    x = x_0 + np.cos(theta) * x_amplitude
    f = f_x(x, *f_args)
    envelope = np.exp(-1j * harmonic * theta)
    return f * envelope


def _sampled_integrand_compiled(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    stack = [
        f_x(x_0 + np.cos(t) * x_amplitude, *f_args) * np.exp(-1j * harmonic * t)
        for t in theta
    ]
    return stack


def _generate_f_theta(f_x):
    def f_theta(theta, x_0, x_amplitude, harmonic, *f_args):
        x = x_0 + x_amplitude * np.cos(theta)
        f = f_x(x, *f_args)
        envelope = np.exp(-1j * harmonic * theta)
        return f * envelope

    return f_theta


def demod(
    f_x,
    x_0,
    x_amplitude,
    harmonic,
    f_args=(),
    method="trapezium",
    n_samples=65,
):
    """Simulate a lock-in amplifier measurement by modulating the input of
    an arbitrary function and demodulating the output.

    For a function `f_x` in the form ``f(x, *args)``, `demod` calculates
    the integral of
    ``f_x(x_0 + x_amplitude * cos(theta)) * exp(-1j * harmonic * theta)``
    for `theta` between -pi and pi.

    Arguments `x_0`, `x_amplitude`, `harmonic` and all `*f_args` should be
    broadcastable according to usual ``numpy`` rules.

    Parameters
    ----------
    f_x : callable
        The function to demodulate. Must be of the form ``f(x, *args)``,
        where `x` is a scalar quantity.
    x_0 : float
        The centre of modulation for the parameter `x`.
    x_amplitude : float
        The modulation amplitude for the parameter `x`.
    harmonic : int
        The harmonic at which to demodulate.
    f_args : tuple
        A tuple of extra arguments to the function `f_x`.
    method : {"trapezium", "simpson", "adaptive"}
        The method to use for the integration. MORE DETAILS HERE.
    n_samples : int
        WRITE ME.

    Returns
    -------
    result : complex
        WRITE ME.

    Notes
    -----

    Examples
    --------
    WRITE ME.
    """
    if method not in ["trapezium", "simpson", "adaptive"]:
        raise ValueError("`method` must be 'trapezium', 'simpson' or 'adaptive'.")

    if method == "adaptive":
        f_theta = _generate_f_theta(f_x)
        result, _ = quad_vec(
            lambda t: f_theta(t, x_0, x_amplitude, harmonic, *f_args), -np.pi, np.pi
        )
        result /= 2 * np.pi
    else:
        si = _sampled_integrand

        integrand = si(f_x, x_0, x_amplitude, harmonic, f_args, n_samples)

        if method == "trapezium":
            result = trapezoid(integrand, axis=0) / (n_samples - 1)
        elif method == "simpson":
            result = simpson(integrand, axis=0) / (n_samples - 1)

    return result
