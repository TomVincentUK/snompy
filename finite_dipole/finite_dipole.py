"""
Finite dipole model (FDM) for predicting contrasts in scanning near-field
optical microscopy (SNOM) measurements.

References
==========
[1] B. Hauer, A.P. Engelhardt, T. Taubner,
    Quasi-analytical model for scattering infrared near-field microscopy on
    layered systems,
    Opt. Express. 20 (2012) 13173.
    https://doi.org/10.1364/OE.20.013173.
[2] A. Cvitkovic, N. Ocelic, R. Hillenbrand
    Analytical model for quantitative prediction of material contrasts in
    scattering-type near-field optical microscopy,
    Opt. Express. 15 (2007) 8550.
    https://doi.org/10.1364/oe.15.008550.
"""
import warnings
import numpy as np
from numba import njit
from scipy.integrate import quad


def complex_quad(func, a, b, **kwargs):
    """
    Wrapper to `scipy.integrate.quad` to allow complex integrands.
    """
    real_part = quad(lambda t, *args: np.real(func(t, *args)), a, b, **kwargs)
    imag_part = quad(lambda t, *args: np.imag(func(t, *args)), a, b, **kwargs)
    return real_part[0] + 1j * imag_part[0], real_part[1] + 1j * imag_part[1]


@njit
def refl_coeff(eps_i, eps_j=1 + 0j):
    """
    Electrostatic reflection coefficient for an interface between materials i
    and j. Defined as beta_ij in equation (7) of reference [1].

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
    return (eps_i - eps_j) / (eps_i + eps_j) + 0j  # + 0j ensures complex


@njit
def geometry_function(z, x, radius, semi_maj_axis, g_factor):
    """
    Function that encapsulates the geometric properties of the tip-sample
    system. Defined as f_0 or f_1 in equations (2) and (11) of reference [1],
    for semi-infinite and multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as H in reference [1].
    x : float
        Position of an induced charge within the tip. Specified relative to the
        tip radius. Defined as W_0 or W_1 in equation (2) of reference [1], and
        X_0 or X_1 in equation (11).
    radius : float
        Radius of curvature of the AFM tip in metres. Defined as rho in
        reference [1].
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as L in reference [1].
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge induced
        in the AFM tip to the magnitude of the nearby charge which induced it.
        A small imaginary component can be used to account for phase shifts
        caused by the capacitive interaction of the tip and sample. Defined as
        g in reference [1].

    Returns
    -------
    f_n : complex
        A complex number encapsulating geometric properties of the tip-sample
        system.
    """
    return (
        (g_factor - (radius + 2 * z + x * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 4 * z + 2 * x * radius))
        / np.log(4 * semi_maj_axis / radius)
    )


@njit
def eff_polarizability(z, beta_0, beta_1, x_0, x_1, radius, semi_maj_axis, g_factor):
    """
    Effective probe-sample polarizability.
    Defined as alpha_eff in equation (3) of reference [1].

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as H in reference [1].
    beta_0 : complex
        Effective electrostatic reflection coefficient for charge 0 of the FDM.
        Defined as beta_0 in equation (11) of reference [1], or simply beta in
        equation (2), where beta_0 and beta_1 are equal.
    beta_1 : complex
        Effective electrostatic reflection coefficient for charge 1 of the FDM.
        Defined as beta_1 in equation (11) of reference [1], or simply beta in
        equation (2), where beta_0 and beta_1 are equal.
    x_0 : float
        Position of induced charge 0 within the tip. Specified relative to the
        tip radius. Defined as W_0 in equation (2) of reference [1], and X_0
        in equation (11).
    x_1 : float
        Position of induced charge 1 within the tip. Specified relative to the
        tip radius. Defined as W_1 in equation (2) of reference [1], and X_1
        in equation (11).
    radius : float
        Radius of curvature of the AFM tip in metres. Defined as rho in
        reference [1].
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as L in reference [1].
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge induced
        in the AFM tip to the magnitude of the nearby charge which induced it.
        A small imaginary component can be used to account for phase shifts
        caused by the capacitive interaction of the tip and sample. Defined as
        g in reference [1].

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.
    """
    f_0 = geometry_function(z, x_0, radius, semi_maj_axis, g_factor)
    f_1 = geometry_function(z, x_1, radius, semi_maj_axis, g_factor)
    return 1 + (beta_0 * f_0) / (2 * (1 - beta_1 * f_1))


@njit
def Fourier_envelope(t, n):
    """
    A complex sinusoid with frequency 2 * pi * `n`, to be used in an integral
    that extracts the nth Fourier coefficient.

    Parameters
    ----------
    t : float
        Domain of the function.
    n : int
        Order of the Fourier component.

    Returns
    -------
    sinusoids : complex
        A complex sinusoid with frequency 2 * pi * `n`.
    """
    return np.exp(-1j * n * t)


@njit
def _integrand(
    t,
    z,
    tapping_amplitude,
    harmonic,
    beta_0,
    beta_1,
    x_0,
    x_1,
    radius,
    semi_maj_axis,
    g_factor,
):
    """
    Function to be integrated from -pi to pi, to extract the Fourier component
    of `eff_polarizability()` corresponding to `harmonic`.
    """
    alpha_eff = eff_polarizability(
        z + tapping_amplitude * (1 + np.cos(t)),
        beta_0,
        beta_1,
        x_0,
        x_1,
        radius,
        semi_maj_axis,
        g_factor,
    )
    sinusoids = Fourier_envelope(t, harmonic)
    return alpha_eff * sinusoids


def _integral(
    z,
    tapping_amplitude,
    harmonic,
    beta_0,
    beta_1,
    x_0,
    x_1,
    radius,
    semi_maj_axis,
    g_factor,
):
    """
    Function that extracts the Fourier component of `eff_polarizability()`
    corresponding to `harmonic`.
    """
    return complex_quad(
        _integrand,
        -np.pi,
        np.pi,
        args=(
            z,
            tapping_amplitude,
            harmonic,
            beta_0,
            beta_1,
            x_0,
            x_1,
            radius,
            semi_maj_axis,
            g_factor,
        ),
    )


# Use this vectorized version instead of `_integral()`.
_integral_vec = np.vectorize(_integral)


def eff_polarizability_nth(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    beta_0=None,
    beta_1=None,
    x_0=1.31,
    x_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    return_err=False,
):
    """
    Effective probe-sample polarizability.
    Defined as alpha_eff in equation (3) of reference [1].

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as H in reference [1].
    tapping_amplitude : float
        The tapping amplitude of the AFM tip. Defined as A in reference [1].
    harmonic : int
        The harmonic of the AFM tip tapping frequency at which to demodulate.
        Defined as n in reference [1].
    eps_sample : complex
        Dielectric function of the sample. Defined as epsilon_s in
        reference [1]. Used to calculate `beta_0`, and ignored if `beta_0` is
        specified.
    beta_0 : complex
        Effective electrostatic reflection coefficient for charge 0 of the FDM.
        Defined as beta_0 in equation (11) of reference [1], or simply beta in
        equation (2), where beta_0 and beta_1 are equal. Overrides `eps_sample`
        if not left as `None`.
    beta_1 : complex
        Effective electrostatic reflection coefficient for charge 1 of the FDM.
        Defined as beta_1 in equation (11) of reference [1], or simply beta in
        equation (2), where beta_0 and beta_1 are equal. If left as `None`,
        `beta_1` takes the same value as `beta_0`, which is used for semi-
        infinite (i.e. not multilayer) samples.
    x_0 : float, default 1.31
        Position of induced charge 0 within the tip. Specified relative to the
        tip radius. Defined as W_0 in equation (2) of reference [1], and X_0
        in equation (11). Default value of 1.31 taken from reference [1].
    x_1 : float, default 0.5
        Position of induced charge 1 within the tip. Specified relative to the
        tip radius. Defined as W_1 in equation (2) of reference [1], and X_1
        in equation (11). Default value of 0.5 taken from reference [1].
    radius : float, default 20e-9
        Radius of curvature of the AFM tip in metres. Defined as rho in
        reference [1].
    semi_maj_axis : float, default 300e-9
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as L in reference [1].
    g_factor : complex, default 0.7 * np.exp(0.06j)
        A dimensionless approximation relating the magnitude of charge induced
        in the AFM tip to the magnitude of the nearby charge which induced it.
        A small imaginary component can be used to account for phase shifts
        caused by the capacitive interaction of the tip and sample. Defined as
        g in reference [1]. Default value of 0.7*e**(0.06j) taken from
        reference [2].
    return_err : bool, default False
        Set to true to return the absolute error in the Fourier integration, as
        estimated by `scipy.integrate.quad`

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.
    alpha_eff_err : complex
        Estimated absolute error from the Fourier integration.
    """
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta_0 is None:
            raise ValueError("Either `eps_sample` or `beta_0` must be specified.")
    else:
        if beta_0 is None:
            beta_0 = refl_coeff(eps_sample)
        else:
            warnings.warn("`beta_0` overrides `eps_sample` when both are specified.")

    # Assume only one beta value unless both specified (for multilayer FDM)
    if beta_1 is None:
        beta_1 = beta_0

    alpha_eff, alpha_eff_err = _integral_vec(
        z,
        tapping_amplitude,
        harmonic,
        beta_0,
        beta_1,
        x_0,
        x_1,
        radius,
        semi_maj_axis,
        g_factor,
    )

    # Normalize to period of integral
    alpha_eff /= 2 * np.pi
    alpha_eff_err /= 2 * np.pi

    if return_err:
        return alpha_eff, alpha_eff_err
    else:
        return alpha_eff
