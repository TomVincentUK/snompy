"""
Finite dipole model (FDM) for predicting contrasts in scanning near-field
optical microscopy (SNOM) measurements.

References
----------
.. [1] B. Hauer, A.P. Engelhardt, T. Taubner,
   Quasi-analytical model for scattering infrared near-field microscopy on
   layered systems,
   Opt. Express. 20 (2012) 13173.
   https://doi.org/10.1364/OE.20.013173.
.. [2] A. Cvitkovic, N. Ocelic, R. Hillenbrand
   Analytical model for quantitative prediction of material contrasts in
   scattering-type near-field optical microscopy,
   Opt. Express. 15 (2007) 8550.
   https://doi.org/10.1364/oe.15.008550.
"""
import numpy as np
from numba import njit

from .tools import complex_quad


@njit
def geom_func_ML(z, x, radius, semi_maj_axis, g_factor):
    """
    Function that encapsulates the geometric properties of the tip-sample
    system. Defined as :math:`f_0` or :math:`f_1` in equation (11) of
    reference [1]_, for multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    x : float
        Position of an induced charge within the tip. Specified relative to
        the tip radius. Defined as :math:`X_0` or :math:`X_1` in equation
        (11) of reference [1]_.
    radius : float
        Radius of curvature of the AFM tip in metres. Defined as
        :math:`\rho` in reference [1]_.
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as :math:`L` in reference [1]_.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample. Defined as :math:`g` in reference [1]_.

    Returns
    -------
    f_n : complex
        A complex number encapsulating geometric properties of the tip-
        sample system.
    """
    return (
        (g_factor - (radius + z + x * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 2 * z + 2 * x * radius))
        / np.log(4 * semi_maj_axis / radius)
    )


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
        A scalar function of momentum, `q`, which returns the complex
        effective reflection coefficient for the stack.
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


@np.vectorize
def potential(z, z_q, beta_q):
    @njit
    def _integrand(xi):
        return beta_q(xi / z_q) * np.exp(-xi * (2 * z_q + z) / z_q)

    integral, error = complex_quad(_integrand, 0, np.inf)
    return integral / z_q, error / z_q


def eff_charge_and_pos(z_q, beta_q, d_z=1e-9):
    phi_0, _ = potential(0, z_q, beta_q)
    phi_p, _ = potential(d_z, z_q, beta_q)
    phi_m, _ = potential(-d_z, z_q, beta_q)
    phi_gradient = (phi_p + phi_m - 2 * phi_0) / d_z**2

    # Need to add in error propagation here too

    X = np.abs(phi_0 / phi_gradient) - z_q
    beta_X = -(phi_0**2) / phi_gradient
    return X, beta_X

@njit
def eff_pol_0_ML(z, beta_q, x_0, x_1, radius, semi_maj_axis, g_factor):
    X_0, beta_0 = eff_charge_and_pos(x_0, beta_q)
    X_1, beta_1 = eff_charge_and_pos(x_1, beta_q)
    f_0 = geom_func_ML(z, X_0, radius, semi_maj_axis, g_factor)
    f_1 = geom_func_ML(z, X_1, radius, semi_maj_axis, g_factor)
    return 1 + (beta_0 * f_0) / (2 * (1 - beta_1 * f_1))
