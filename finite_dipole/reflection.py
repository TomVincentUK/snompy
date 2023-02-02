"""
Reflection coefficients (:mod:`finite_dipole.reflection`)
=========================================================

.. currentmodule:: finite_dipole.reflection

This module provides functions for calculating reflection coefficients of
interfaces between two materials, or the momentum-dependent effective
reflection coefficient for a stack of three or more materials.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    refl_coeff
    refl_coeff_ML
    interface_stack
"""
import warnings

import numpy as np


def refl_coeff(eps_i, eps_j):
    """Return the electrostatic reflection coefficient for an interface
    between two materials.

    Calculated using ``(eps_j - eps_i) / (eps_j + eps_i)``, where `eps_i`
    and `eps_j` are the dielectric functions of two materials `i` and `j`.

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

    Examples
    --------
    Works with real inputs:

    >>> refl_coeff(1, 3)
    0.5

    Works with complex inputs:

    >>> refl_coeff(1, 1 + 1j)
    (0.2+0.4j)

    Performs vectorised operations:

    >>> refl_coeff([1, 3], [3, 5])
    array([0.5 , 0.25])
    >>> refl_coeff([1, 3], [[1], [3]])
    array([[ 0. , -0.5],
          [ 0.5,  0. ]])
    """
    eps_i = np.asarray(eps_i)
    eps_j = np.asarray(eps_j)
    return (eps_j - eps_i) / (eps_j + eps_i)


def refl_coeff_ML(k, beta_stack, t_stack):
    r"""Return the momentum-dependent effective reflection coefficient for
    the first interface in a stack of interfaces.

    Parameters
    ----------
    k : float or array_like
        In-plane electromagnetic wave momentum. If `k` is array_like, it
        must be broadcastable with `beta_stack[0]` and `t_stack[0]`.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        `beta_stack[0]` must be broadcastable with `k` and `t_stack[0]`.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the interfaces in
        `beta_stack`. Must have first dimension size one fewer than
        `beta_stack`. `t_stack[0]` must be broadcastable with `k` and
        `beta_stack[0]`. A zero-size array can be used for the case of a
        single interface.

    Returns
    -------
    beta_k : complex
        The momentum-dependent effective reflection coefficient.

    Notes
    -----
    This function works by recursively applying the equation

    .. math::

        \beta(k) =
        \frac{\beta_{01} + \beta_{12}e^{-2kt_1}}
        {1 + \beta_{01}\beta_{12}e^{-2kt_1}}

    as an expression for :math:`\beta_{12}`, where :math:`\beta_{ij}` is
    the electrostatic reflection coefficient between layers :math:`i` and
    :math:`j`, :math:`t_{i}` is the thickness of the :math:`i^{th}`
    layer, and :math:`k` is the in-plane momentum_[1].

    References
    ----------
    .. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
       R. Hillenbrand, “Subsurface chemical nanoidentification by nano-FTIR
       spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359, Dec. 2020,
       doi: 10.1038/s41467-020-17034-6.

    Examples
    --------

    Momentum-dependent result for multiple interfaces:

    >>> import numpy as np
    >>> k = np.arange(5)
    >>> beta_stack = np.array([1 / 3, 1 / 5])
    >>> t_stack = np.array([1])
    >>> refl_coeff_ML(k, beta_stack, t_stack)
    array([0.5  , 0.357, 0.337, 0.333, 0.333])

    Constant value for single interface:

    >>> k = np.arange(5)
    >>> refl_coeff_ML(k, beta_stack=np.array([0.5]), t_stack=np.array([]))
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    """
    beta_effective = beta_stack[0] * np.ones_like(k)
    for i in range(t_stack.shape[0]):
        layer_decay = np.exp(-2 * k * t_stack[i])
        beta_effective = (beta_effective + beta_stack[i + 1] * layer_decay) / (
            1 + beta_effective * beta_stack[i + 1] * layer_decay
        )
    return beta_effective


def interface_stack(eps_stack=None, beta_stack=None, t_stack=None):
    r"""Return a stack of reflection coefficients and layer thicknesses in
    the form required by `refl_coeff_ML`.

    Parameters
    ----------
    eps_stack : array_like
        Dielectric functions of each layer in the stack. Layers should be
        arranged from the top down, starting with the semi-infinite
        superstrate and ending with the semi-infinite substrate. Ignored
        if `beta_stack` is specified.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.

    Returns
    -------
    beta_stack : np.ndarray
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
    t_stack : np.ndarray
        Thicknesses of each sandwiched layer between the interfaces in
        `beta_stack`. A zero-size array is returned for the case of a
        single interface.

    Examples
    --------
    Ensures outputs are in form of `numpy` arrays:

    >>> interface_stack(beta_stack=(0.75, 0.5, 0.25), t_stack=(100e-9, 50e-9))
    (array([0.75, 0.5 , 0.25]), array([1.e-07, 5.e-08]))

    Converts `eps_stack` to `beta_stack`:

    >>> interface_stack(eps_stack=(1, 2, 3, 4), t_stack=(100e-9, 50e-9))
    (array([0.33, 0.2 , 0.14]), array([1.e-07, 5.e-08]))

    Ensures output arrays are non-ragged:

    >>> ragged_beta = (0.75, (0.5, 0.25), 0.25)
    >>> interface_stack(beta_stack=ragged_beta, t_stack=(100e-9, 50e-9))[0]
    array([[0.75, 0.75],
           [0.5 , 0.25],
           [0.25, 0.25]])

    Works for single interfaces:

    >>> interface_stack(beta_stack=(0.5,))
    (array([0.5]), array([], dtype=float64))
    >>> interface_stack(eps_stack=(1, 2))
    (array([0.33]), array([], dtype=float64))
    >>> interface_stack(beta_stack=(0.5,), t_stack=())
    (array([0.5]), array([], dtype=float64))
    """
    if t_stack is None:
        t_stack = np.array([])
    else:
        t_stack = np.asarray(np.broadcast_arrays(*t_stack))

    if (t_stack == 0).any():
        warnings.warn(
            " ".join(
                "`t_stack` contains zeros.",
                "Zero-thickness dielectric layers are unphysical.",
                "Results may not be as expected.",
            )
        )

    if eps_stack is None:
        if beta_stack is None:
            raise ValueError("Either `eps_stack` or `beta_stack` must be specified.")
        beta_stack = np.asarray(np.broadcast_arrays(*beta_stack))
    else:
        if beta_stack is None:
            # beta_stack calculated from eps_stack if not specified
            eps_stack = np.asarray(np.broadcast_arrays(*eps_stack))
            if eps_stack.shape[0] != t_stack.shape[0] + 2:
                raise ValueError(
                    "`eps_stack` must be 2 longer than `t_stack` along the first axis."
                )
            beta_stack = refl_coeff(eps_stack[:-1], eps_stack[1:])
        else:
            warnings.warn("`beta_stack` overrides `eps_stack` when both are specified.")
            beta_stack = np.asarray(np.broadcast_arrays(*beta_stack))

    if beta_stack.shape[0] != t_stack.shape[0] + 1:
        raise ValueError(
            "`beta_stack` must be 1 longer than `t_stack` along the first axis."
        )

    return beta_stack, t_stack
