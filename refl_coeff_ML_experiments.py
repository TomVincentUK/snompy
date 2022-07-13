import numpy as np
from numba import njit

from finite_dipole.tools import refl_coeff


def refl_coeff_ML(beta_stack, t_stack):
    """
    Calculates the momentum-dependent effective reflection coefficient for
    the first interface in a stack of layers sandwiched between a semi-
    infinite superstrate and substrate.

    Parameters
    ----------
    beta_stack : array like
        Reflection coefficients of each interface in the stack (with the
        first element corresponding to the top interface).
    t_stack : float
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one less than the
        number of interfaces.

    Returns
    -------
    beta_q : function
        A scalar function of momentum, `q`, which returns the effective
        reflection coefficient for the stack.
    """
    if len(beta_stack) != len(t_stack) + 1:
        raise ValueError("`beta_stack` must have length 1 greater than `t_stack`.")

    if len(beta_stack) == 1:
        beta_final = beta_stack[0]

        @njit
        def beta_q(q):
            return beta_final

    else:
        beta_current = beta_stack[0]
        t_current = t_stack[0]

        beta_stack_next = beta_stack[1:]
        t_stack_next = t_stack[1:]
        beta_next = refl_coeff_ML(beta_stack_next, t_stack_next)

        @njit
        def beta_q(q):
            next_layer = beta_next(q) * np.exp(-2 * q * t_current)
            return (beta_current + next_layer) / (1 + beta_current * next_layer)

    return beta_q


thickness = 10e-9
eps_stack = 1, 0.1 + 0.1j, 6 - 12j
beta_stack = refl_coeff(eps_stack[:-1], eps_stack[1:])
beta_q = refl_coeff_ML(beta_stack, [thickness])
