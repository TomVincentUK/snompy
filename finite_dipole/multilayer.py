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

from .demodulate import demod
from .reflection import beta_and_t_stack_from_inputs, refl_coeff_ML

# Default values
N_LAG = 64


def phi_E_0(z_q, beta_stack, t_stack, N_Lag=N_LAG):
    """Calculate phi and E using Gauss-Laguerre quadrature"""
    # Evaluate integral in terms of x = k * 2 * z_q
    x_Lag, w_Lag = np.polynomial.laguerre.laggauss(N_Lag)
    k = x_Lag / np.asarray(2 * z_q)[..., np.newaxis]

    beta_k = refl_coeff_ML(k, beta_stack[..., np.newaxis], t_stack[..., np.newaxis])

    phi = np.sum(w_Lag * beta_k, axis=-1) / (2 * z_q)
    E = np.sum(w_Lag * x_Lag * beta_k, axis=-1) / (4 * z_q**2)

    return phi, E


def eff_pos_and_charge(z_q, beta_stack, t_stack, N_Lag=N_LAG):
    phi, E = phi_E_0(z_q, beta_stack, t_stack, N_Lag)
    z_image = np.abs(phi / E) - z_q
    beta_image = phi**2 / E
    return z_image, beta_image


def geom_func_ML(z, z_image, radius, semi_maj_axis, g_factor):
    """Function that encapsulates the geometric properties of the tip-
    sample system. Defined as `f_0` or `f_1` in equation (11) of reference
    [1], for multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as `H` in
        reference [1].
    z_image : float
        Position of an image charge below the surface.
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
        (g_factor - (radius + z + z_image) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 2 * z + 2 * z_image))
        / np.log(4 * semi_maj_axis / radius)
    )


def eff_pol_0_ML(
    z,
    beta_stack=None,
    t_stack=None,
    x_0=1.31 * 15 / 16,
    x_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    N_Lag=N_LAG,
):
    z_q_0 = z + radius * x_0
    z_im_0, beta_im_0 = eff_pos_and_charge(z_q_0, beta_stack, t_stack, N_Lag)
    f_0 = geom_func_ML(z, z_im_0, radius, semi_maj_axis, g_factor)

    z_q_1 = z + radius * x_1
    z_im_1, beta_im_1 = eff_pos_and_charge(z_q_1, beta_stack, t_stack, N_Lag)
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
    N_Lag=N_LAG,
):
    if x_0 is None:
        x_0 = 1.31 * semi_maj_axis / (semi_maj_axis + 2 * radius)

    beta_stack, t_stack = beta_and_t_stack_from_inputs(eps_stack, beta_stack, t_stack)

    # Set oscillation centre so AFM tip touches sample at z = 0
    z_0 = z + tapping_amplitude + radius
    alpha_eff = demod(
        eff_pol_0_ML,
        z_0,
        tapping_amplitude,
        harmonic,
        f_args=(
            beta_stack,
            t_stack,
            x_0,
            x_1,
            radius,
            semi_maj_axis,
            g_factor,
            N_Lag,
        ),
        method=demod_method,
    )

    return alpha_eff
