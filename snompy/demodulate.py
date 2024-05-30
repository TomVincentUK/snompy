"""
Demodulation (:mod:`snompy.demodulate`)
=======================================

.. currentmodule:: snompy.demodulate

This module provides a function to simulate lock-in amplifier measurements
of arbitrary functions.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    demod
"""

import numpy as np

from ._defaults import defaults
from ._utils import _pad_for_broadcasting


def demod(f_x, x_0, x_amplitude, n, f_args=(), n_trapz=None, **kwargs):
    r"""Simulate a lock-in amplifier measurement by modulating the input of
    an arbitrary function then demodulating the output.

    This function multiplies the modulated output of `f_x` by a complex
    envelope, which has a period of `n` times pi, then integrates
    between -pi and pi, using the trapezium method.

    Parameters
    ----------
    f_x : callable
        The function to demodulate. Must be of the form
        ``f(x, *args, **kwargs)``, where `x` is a scalar quantity.
    x_0 : float
        The centre of modulation for the parameter `x`.
    x_amplitude : float
        The modulation amplitude for the parameter `x`.
    n : int
        The harmonic at which to demodulate.
    f_args : tuple
        A tuple of extra positional arguments to the function `f_x`.
    n_trapz : int
        The number of intervals to use for the trapezium-method
        integration.
    **kwargs : dict, optional
        Extra keyword arguments are passed to the function `f_x`.

    Returns
    -------
    result : complex
        The modulated and demodulated output.

    Notes
    -----
    For a function in the form `f_x(x, *args, **kwargs)`, :func:`demod` calculates

    .. math::

        \frac{1}{2 \pi}
        \int_{-\pi}^{\pi}
        \mathtt{f\_x(x\_0} + \mathtt{x\_amplitude} \cdot \cos(\theta)
        \mathtt{, *args, **kwargs)}
        e^{-i \theta \cdot \mathtt{n}}
        d{\theta}

    using the trapezium method.

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
    >>> n = np.arange(3)[:, np.newaxis, np.newaxis]
    >>> y = np.arange(4)[:, np.newaxis, np.newaxis, np.newaxis]
    >>> demod(lambda x, y: x * y, x_0, x_amplitude, n, (y,)).shape
    (4, 3, 2, 1)
    """
    x_0 = np.asanyarray(x_0)
    x_amplitude = np.asanyarray(x_amplitude)
    n = np.asanyarray(n)
    n_trapz = defaults.n_trapz if n_trapz is None else n_trapz

    theta = _pad_for_broadcasting(
        np.linspace(-np.pi, np.pi, n_trapz + 1),
        (f_x(x_0 + 0 * x_amplitude, *f_args, **kwargs), n),
    )
    integrand = f_x(x_0 + np.cos(theta) * x_amplitude, *f_args, **kwargs) * np.exp(
        -1j * n * theta
    )
    return np.trapz(integrand, axis=0) / (n_trapz)
