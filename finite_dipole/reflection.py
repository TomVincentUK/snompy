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
"""
import warnings

import numpy as np
from numba import njit, vectorize


@vectorize(["float64(float64, float64)", "complex128(complex128, complex128)"])
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

    Copes with vectorised operations:

    >>> refl_coeff([1, 3], [3, 5])
    array([0.5 , 0.25])
    >>> refl_coeff([1, 3], [[1], [3]])
    array([[ 0. , -0.5],
          [ 0.5,  0. ]])
    """
    return (eps_j - eps_i) / (eps_j + eps_i)


@njit(cache=True)
def refl_coeff_ML(k, beta_stack, t_stack):
    """Write me."""
    beta_effective = beta_stack[0] * np.ones_like(k * t_stack[0])
    for i in range(t_stack.shape[0]):
        layer_decay = np.exp(-2 * k * t_stack[i])
        beta_effective = (beta_effective + beta_stack[i + 1] * layer_decay) / (
            1 + beta_effective * beta_stack[i + 1] * layer_decay
        )
    return beta_effective


def beta_and_t_stack_from_inputs(eps_stack=None, beta_stack=None, t_stack=None):
    """Write me."""
    if t_stack is None:
        t_stack = np.array([])
    else:
        t_stack = np.asarray(np.broadcast_arrays(*t_stack))

    if (t_stack == 0).any():
        warnings.warn(
            " ".join(
                "`t_stack` contains 0 values.",
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
