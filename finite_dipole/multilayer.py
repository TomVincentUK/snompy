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
import warnings

import numpy as np
from numba import njit, vectorize
from scipy.integrate import quad_vec

from .demodulate import demod
from .tools import refl_coeff


def _beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack):
    if t_stack is None:
        t_stack = np.array([])
    else:
        t_stack = np.asarray(t_stack)

    if eps_stack is None:
        if beta_stack is None:
            raise ValueError("Either `eps_stack` or `beta_stack` must be specified.")
        beta_stack = np.asarray(beta_stack)
    else:
        if beta_stack is None:
            # beta_stack calculated from eps_stack if not specified
            eps_stack = np.asarray(eps_stack)
            if eps_stack.shape[0] != t_stack.shape[0] + 2:
                raise ValueError(
                    "`eps_stack` must be 2 longer than `t_stack` along the first axis."
                )
            beta_stack = refl_coeff(eps_stack[:-1], eps_stack[1:])
        else:
            warnings.warn("`beta_stack` overrides `eps_stack` when both are specified.")
            beta_stack = np.asarray(beta_stack)

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
    return beta_k


@njit
def _phi_k_weighting(k, z_q):
    """Used by potential_0()"""
    return np.exp(-k * 2 * z_q)


def potential_0(z_q, beta_k):
    """
    Potential induced at z=0 by a charge q, at height `z_q`, and its image
    charge, over an interface with momentum-dependent reflection
    coefficient `beta_k(k)`.
    """

    @njit
    def integrand(xi):
        k = xi / z_q  # xi is just k adjusted to the characteristic length scale
        return beta_k(k) * _phi_k_weighting(k, z_q)

    integral, _ = quad_vec(integrand, 0, np.inf)

    # Rescale from xi to k
    integral /= z_q
    return integral


@njit
def _E_k_weighting(k, z_q):
    """Used by E_z_0()"""
    return k * np.exp(-k * 2 * z_q)


def E_z_0(z_q, beta_k):
    """
    z-component of the electric field induced at z=0 by a charge q, at
    height `z_q`, and its image charge, over an interface with momentum-
    dependent reflection coefficient `beta_k(k)`.
    """

    @njit
    def integrand(xi):
        k = xi / z_q  # xi is just k adjusted to the characteristic length scale
        return beta_k(k) * _E_k_weighting(k, z_q)

    integral, _ = quad_vec(integrand, 0, np.inf)

    # Rescale from xi to k
    integral /= z_q
    return integral


def eff_charge_and_pos(z_q, beta_k):
    phi = potential_0(z_q, beta_k)
    E = E_z_0(z_q, beta_k)

    z_image = np.abs(phi / E) - z_q
    beta_image = phi**2 / E
    return z_image, beta_image


@njit
def geom_func_ML(z, z_q, radius, semi_maj_axis, g_factor):
    """
    Function that encapsulates the geometric properties of the tip-sample
    system. Defined as :math:`f_0` or :math:`f_1` in equation (11) of
    reference [1]_, for multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    z_q : float
        Position of an image charge below the surface. Defined as
        :math:`X_0` or :math:`X_1` in equation (11) of reference [1]_.
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
        (g_factor - (radius + z + z_q) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 2 * z + 2 * z_q))
        / np.log(4 * semi_maj_axis / radius)
    )


def eff_pol_0_ML(z, beta_k, x_0, x_1, radius, semi_maj_axis, g_factor):
    z_q_0 = z + radius * x_0
    z_im_0, beta_im_0 = eff_charge_and_pos(z_q_0, beta_k)
    f_0 = geom_func_ML(z, z_im_0, radius, semi_maj_axis, g_factor)

    z_q_1 = z + radius * x_1
    z_im_1, beta_im_1 = eff_charge_and_pos(z_q_1, beta_k)
    f_1 = geom_func_ML(z, z_im_1, radius, semi_maj_axis, g_factor)

    return 1 + (beta_im_0 * f_0) / (2 * (1 - beta_im_1 * f_1))


def eff_pol_ML(
    z,
    tapping_amplitude,
    harmonic,
    eps_stack=None,
    beta_stack=None,
    t_stack=None,
    x_0=None,
    x_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    demod_method="trapezium",
):
    if x_0 is None:
        x_0 = 1.31 * semi_maj_axis / (semi_maj_axis + 2 * radius)

    beta_k = refl_coeff_ML(eps_stack, beta_stack, t_stack)

    alpha_eff = demod(
        eff_pol_0_ML,
        z + tapping_amplitude,  # add the amplitude so z_0 is at centre of oscillation
        tapping_amplitude,
        harmonic,
        f_args=(beta_k, x_0, x_1, radius, semi_maj_axis, g_factor),
        method=demod_method,
    )

    return alpha_eff
