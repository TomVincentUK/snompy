"""Functions for calculating reflection coefficients.
"""
import warnings

import numpy as np
from numba import njit, vectorize


@vectorize(["float64(float64, float64)", "complex128(complex128, complex128)"])
def refl_coeff(eps_i, eps_j):
    """Electrostatic reflection coefficient for an interface between
    materials `i` and `j`.

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


def _beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack):
    """Function to convert stacks of dielectric functions, reflection
    coefficients and layer thicknesses to a stack of reflection
    coefficients and layer thicknesses in the desired format for
    _beta_func_from_stack.
    """
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
    """Recursive function which returns `beta_k(k)`, the momentum-dependent
    effective reflection coefficient for a stack of interfaces with
    electrostatic reflection coefficients `beta_stack` and interface
    separations `t_stack`.
    """
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
    r"""Calculates the momentum-dependent effective reflection coefficient
    for the first interface in a stack of layers sandwiched between a semi-
    infinite superstrate and substrate.

    Parameters
    ----------
    eps_stack : array like
        Dielectric functions of each layer in the stack. Layers should be
        arranged from the top down, starting with the semi-infinite
        superstrate and ending with the semi-infinite substrate. Ignored
        if `beta_stack` is specified.
    beta_stack : array like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified
    t_stack : float
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one less than
        `beta_stack` or two less than `eps_stack`. An empty list can be
        used for the case of a single interface.

    Returns
    -------
    beta_k : function
        A scalar function of momentum, `k`, which returns the complex
        effective reflection coefficient for the stack.

    Notes
    -----
    This function works by recursively applying the equation

    .. math::

        \beta(k) =
        \frac{\beta_{01} + \beta_{12}e^{-2kt_1}}
        {1 + \beta_{01}\beta_{12}e^{-2kt_1}}

    as an expression for :math:`\beta_{12}`, where :math:`\beta_{ij}` is
    the electrostatic reflection coefficient between layers :math:`i` and
    :math:`j`, and :math:`t_{i}` is the thickness of the :math:`i^{th}`
    layer._[1]

    References
    ----------
    .. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
       R. Hillenbrand, “Subsurface chemical nanoidentification by nano-FTIR
       spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359, Dec. 2020,
       doi: 10.1038/s41467-020-17034-6.

    Examples
    --------
    Dielectric function specified:

    >>> beta_k = refl_coeff_ML(eps_stack=(1, 2, 3), t_stack=(1,))
    >>> [beta_k(k) for k in range(5)]
    [0.500, 0.357, 0.337, 0.334, 0.333]

    Reflection coefficients specified:

    >>> beta_k = refl_coeff_ML(beta_stack=(1 / 3, 1 / 5), t_stack=(1,))
    >>> [beta_k(k) for k in range(5)]
    [0.500, 0.357, 0.337, 0.334, 0.333]

    Complex inputs:

    >>> beta_k = refl_coeff_ML(eps_stack=(1, 1 + 1j, 3), t_stack=(1,))
    >>> [beta_k(k) for k in range(4)]
    [(0.500+0.000j), (0.252+0.339j), (0.207+0.392j), (0.201+0.399j)]

    Non-scalar inputs:

    >>> eps_stack = 1, (2, 3, 4, 5, 6, 7, 8), 9
    >>> beta_k = refl_coeff_ML(eps_stack=eps_stack, t_stack=(1,))
    >>> beta_k(0)
    array([0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800])

    Single interface:

    >>> beta_k = refl_coeff_ML(beta_stack=(0.5,))
    >>> [beta_k(k) for k in range(5)]
    [0.500, 0.500, 0.500, 0.500, 0.500]
    """
    beta_stack, t_stack = _beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack)
    beta_k = _beta_func_from_stack(beta_stack, t_stack)

    beta_k.ndim = np.ndim(
        beta_k(0)
    )  # Ensures np.ndim(beta_k) = np.ndim(beta_k(scalar))

    return beta_k
