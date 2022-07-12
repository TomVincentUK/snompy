import numpy as np
from numba import njit


def refl_coeff_ML(beta_stack, t_stack):
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


def interface_refl_coeffs(eps_stack):
    eps_stack = np.asarray(eps_stack)
    return (eps_stack[1:] - eps_stack[:-1]) / (eps_stack[1:] + eps_stack[:-1])


thickness = 10e-9
eps_stack = 1, 0.1 + 0.1j, 6 - 12j
beta_stack = interface_refl_coeffs(eps_stack)
beta_q = refl_coeff_ML(beta_stack, [thickness])
