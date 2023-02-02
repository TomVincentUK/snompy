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

from ._defaults import defaults


def demod(
    f_x,
    x_0,
    x_amplitude,
    harmonic,
    f_args=(),
    N_demod_trapz=defaults["N_demod_trapz"],
):
    """Simulate a lock-in amplifier measurement by modulating the input of
    an arbitrary function then demodulating the output.

    For a function `f_x` in the form ``f(x, *args)``, `demod` calculates
    the integral of
    ``f_x(x_0 + x_amplitude * cos(theta)) * exp(-1j * harmonic * theta)``
    for `theta` between -pi and pi, using the trapezium method.


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
    N_demod_trapz : int
        The number of intervals to use for the trapezium-method
        integration.

    Returns
    -------
    result : complex
        The modulated and demodulated output.

    Examples
    --------
    Works with complex functions:

    >>> demod(lambda x: (1 + 2j) * x**2, 0, 1, 2)
    (0.25+0.50j)

    Accepts extra arguments to `f_x`:

    >>> demod(lambda x, m, c: m * x + c, 0, 1, 1, f_args=(1, 1j))
    (0.5+0.0j)

    Broadcastable inputs:

    >>> import numpy as np
    >>> x_0 = 1
    >>> x_amplitude = np.arange(2)[:, np.newaxis]
    >>> harmonic = np.arange(3)[:, np.newaxis, np.newaxis]
    >>> y = np.arange(4)[:, np.newaxis, np.newaxis, np.newaxis]
    >>> demod(lambda x, y: x * y, x_0, x_amplitude, harmonic, (y,)).shape
    (4, 3, 2, 1)
    """
    output_ndim = np.asarray(f_x(x_0 + 0 * x_amplitude, *f_args) * harmonic).ndim
    theta = np.linspace(-np.pi, np.pi, N_demod_trapz + 1).reshape(
        -1, *(1,) * output_ndim
    )
    integrand = f_x(x_0 + np.cos(theta) * x_amplitude, *f_args) * np.exp(
        -1j * harmonic * theta
    )
    return np.trapz(integrand, axis=0) / N_demod_trapz
