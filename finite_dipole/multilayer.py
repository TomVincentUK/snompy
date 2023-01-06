"""
Multilayer finite dipole model (:mod:`finite_dipole.multilayer`)
================================================================

.. currentmodule:: finite_dipole.multilayer

WRITE A DESCRIPTION HERE.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    potential_0
    E_z_0
    geom_func_ML
    eff_pol_0_ML
    eff_pol_ML
"""
import numpy as np
from numba import njit
from scipy.integrate import quad_vec

from .demodulate import demod
from .reflection import refl_coeff_ML


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
    """Function that encapsulates the geometric properties of the tip-
    sample system. Defined as `f_0` or `f_1` in equation (11) of reference
    [1], for multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as `H` in
        reference [1].
    z_q : float
        Position of an image charge below the surface. Defined as
        `X_0` or `X_1` in equation (11) of reference [1].
    radius : float
        Radius of curvature of the AFM tip in metres. Defined as
        `rho` in reference [1].
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as `L` in reference [1].
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample. Defined as `g` in reference [1].

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
