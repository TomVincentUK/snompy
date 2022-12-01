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
from numba import njit

from .tools import refl_coeff
from .demodulate import demod


@njit
def geom_func(z, x, radius, semi_maj_axis, g_factor):
    """
    Function that encapsulates the geometric properties of the tip-sample
    system. Defined as :math:`f_0` or :math:`f_1` in equation (2) of
    reference [1]_, for semi-infinite samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    x : float
        Position of an induced charge within the tip. Specified relative to
        the tip radius. Defined as :math:`W_0` or :math:`W_1` in equation
        (2) of reference [1]_.
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
        (g_factor - (radius + 2 * z + x * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 4 * z + 2 * x * radius))
        / np.log(4 * semi_maj_axis / radius)
    )


@njit
def eff_pol_0(z, beta, x_0, x_1, radius, semi_maj_axis, g_factor):
    """
    Effective probe-sample polarizability.
    Defined as :math:`\alpha_{eff}`` in equation (3) of reference [1]_.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    beta : complex
        Effective electrostatic reflection coefficient the interface.
        Defined as :math:`\beta` in equation (2) of reference [1]_.
    x_0 : float
        Position of induced charge 0 within the tip. Specified relative to
        the tip radius. Defined as :math:`W_0` in equation (2) of reference
        [1]_.
    x_1 : float
        Position of induced charge 1 within the tip. Specified relative to
        the tip radius. Defined as :math:`W_1` in equation (2) of reference
        [1]_.
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
    alpha_eff : complex
        Effective polarizability of the tip and sample.
    """
    f_0 = geom_func(z, x_0, radius, semi_maj_axis, g_factor)
    f_1 = geom_func(z, x_1, radius, semi_maj_axis, g_factor)
    return 1 + (beta * f_0) / (2 * (1 - beta * f_1))


def eff_pol(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    beta=None,
    x_0=None,
    x_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    demod_method="trapezium",
):
    """
    Effective probe-sample polarizability.
    Defined as :math:`\alpha_{eff, n}` in reference [1]_.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    tapping_amplitude : float
        The tapping amplitude of the AFM tip. Defined as :math:`A` in
        reference [1]_.
    harmonic : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate. Defined as :math:`n` in reference [1]_.
    eps_sample : complex
        Dielectric function of the sample. Defined as :math:`\epsilon_s` in
        reference [1]_. Used to calculate `beta_0`, and ignored if `beta_0`
        is specified.
    beta : complex
        Effective electrostatic reflection coefficient the interface.
        Defined as :math:`\beta` in equation (2) of reference [1]_.
    x_0 : float
        Position of induced charge 0 within the tip. Specified relative to
        the tip radius. Defined as :math:`W_0` in equation (2) of reference
        [1]_, and :math:`X_0` in equation (11).
    x_1 : float
        Position of induced charge 1 within the tip. Specified relative to
        the tip radius. Defined as :math:`W_1` in equation (2) of reference
        [1]_, and :math:`X_1` in equation (11).
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
        sample. Defined as :math:`g` in reference [1]_. Default value of
        :math:`0.7 e^{0.06i}`` taken from reference [2]_.
    demod_method : {'trapezium', 'simpson' or 'adaptive'}
        Integration method used to demodulate the signal.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.
    alpha_eff_err : complex
        Estimated absolute error from the Fourier integration.
    """
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta is None:
            raise ValueError("Either `eps_sample` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(1 + 0j, eps_sample)
        else:
            warnings.warn("`beta` overrides `eps_sample` when both are specified.")

    if x_0 is None:
        x_0 = 1.31 * semi_maj_axis / (semi_maj_axis + 2 * radius)

    alpha_eff = demod(
        eff_pol_0,
        z + tapping_amplitude,  # add the amplitude so z_0 is at centre of oscillation
        tapping_amplitude,
        harmonic,
        f_args=(beta, x_0, x_1, radius, semi_maj_axis, g_factor),
        method=demod_method,
    )

    return alpha_eff
