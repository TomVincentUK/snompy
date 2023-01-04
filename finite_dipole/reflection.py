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
import warnings

import numpy as np
from numba import njit, vectorize


@vectorize(["float64(float64, float64)", "complex128(complex128, complex128)"])
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
    return (eps_j - eps_i) / (eps_j + eps_i)


def _beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack):
    if t_stack is None:
        t_stack = np.array([])
    else:
        t_stack = np.asarray(np.broadcast_arrays(*t_stack))

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


def _beta_func_from_stack(beta_stack, t_stack):
    if beta_stack.shape[0] == 1:
        beta_final = beta_stack[0]

        @njit
        def beta_k(_):
            return beta_final

    else:
        beta_current = beta_stack[0]
        t_current = t_stack[0]

        beta_stack_next = beta_stack[1:]
        t_stack_next = t_stack[1:]
        beta_next = _beta_func_from_stack(beta_stack_next, t_stack_next)

        @njit
        def beta_k(k):
            next_layer = beta_next(k) * np.exp(-2 * k * t_current)
            return (beta_current + next_layer) / (1 + beta_current * next_layer)

    return beta_k


def refl_coeff_ML(eps_stack=None, beta_stack=None, t_stack=None):
    """
    Calculates the momentum-dependent effective reflection coefficient for
    the first interface in a stack of layers sandwiched between a semi-
    infinite superstrate and substrate.

    Parameters
    ----------
    eps_stack : array like
        WRITE ME.
    beta_stack : array like
        Reflection coefficients of each interface in the stack (with the
        first element corresponding to the top interface).
    t_stack : float
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one less than the
        number of interfaces.

    Returns
    -------
    beta_k : function
        A scalar function of momentum, `q`, which returns the complex
        effective reflection coefficient for the stack.
    """
    beta_stack, t_stack = _beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack)
    beta_k = _beta_func_from_stack(beta_stack, t_stack)

    beta_k.ndim = np.ndim(
        beta_k(0)
    )  # Ensures np.ndim(beta_k) = np.ndim(beta_k(scalar))

    return beta_k
