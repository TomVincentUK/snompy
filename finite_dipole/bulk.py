"""
Bulk finite dipole model (:mod:`finite_dipole.bulk`)
================================================================

.. currentmodule:: finite_dipole.bulk

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the bulk finite
dipole model (FDM) for semi-infinite substrate and superstrates.


.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol
    eff_pol_0
    geom_func
"""
import warnings

import numpy as np

from ._defaults import defaults
from .demodulate import demod
from .reflection import refl_coeff


def geom_func(
    z,
    x,
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for bulk finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    x : float
        Position of an induced charge within the tip. Specified in units of
        the tip radius.
    radius : float
        Radius of curvature of the AFM tip.
    semi_maj_axis : float
        Semi-major axis length of the effective spheroid from the finite
        dipole model.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.

    Returns
    -------
    f_n : complex
        A complex number encapsulating geometric properties of the tip-
        sample system.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{geom} =
        \left(
            g - \frac{r + 2 z + W}{2 L}
        \right)
        \frac{\ln{\left(\frac{4 L}{r + 4 z + 2 W}\right)}}
        {\ln{\left(\frac{4 L}{r}\right)}}

    where :math:`z` is `z`, :math:`W` is `x * radius`, :math:`r` is
    `radius`, :math:`L` is `semi_maj_axis`, and :math:`g` is `g_factor`.
    This is given as equation (2) in reference_[1].

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    return (
        (g_factor - (radius + 2 * z + x * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 4 * z + 2 * x * radius))
        / np.log(4 * semi_maj_axis / radius)
    )


def eff_pol_0(
    z,
    beta,
    x_0=defaults["x_0"],
    x_1=defaults["x_1"],
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
):
    r"""Return the effective probe-sample polarizability using the bulk
    finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
    x_0 : float
        Position of an induced charge 0 within the tip. Specified in units
        of the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified in units
        of the tip radius.
    radius : float
        Radius of curvature of the AFM tip.
    semi_maj_axis : float
        Semi-major axis length of the effective spheroid from the finite
        dipole model.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} =
        1
        + \frac{\beta f_{geom}(z, x_0, r, L, g)}
        {2 (1 - \beta f_{geom}(z, x_1, r, L, g))}

    where :math:`\alpha_{eff}` is `\alpha_eff`, :math:`\beta` is `beta`,
    :math:`r` is `radius`, :math:`L` is `semi_maj_axis`, :math:`g` is
    `g_factor`, and :math:`f_{geom}` is a function encapsulating the
    geometric properties of the tip-sample system. This is given as
    equation (3) in reference_[1]. The function :math:`f_{geom}` is
    implemented here as `finite_dipole.bulk.geom_func`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    f_0 = geom_func(z, x_0, radius, semi_maj_axis, g_factor)
    f_1 = geom_func(z, x_1, radius, semi_maj_axis, g_factor)
    return 1 + (beta * f_0) / (2 * (1 - beta * f_1))


def eff_pol(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    eps_environment=defaults["eps_environment"],
    beta=None,
    x_0=None,
    x_1=defaults["x_1"],
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the bulk finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    tapping_amplitude : float
        The tapping amplitude of the AFM tip.
    harmonic : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    eps_sample : complex
        Dielectric function of the sample. Used to calculate `beta_0`, and
        ignored if `beta_0` is specified.
    eps_environment : complex
        Dielectric function of the environment (superstrate). Used to
        calculate `beta_0`, and ignored if `beta_0` is specified.
    beta : complex
        Electrostatic reflection coefficient of the interface.
    x_0 : float
        Position of an induced charge 0 within the tip. Specified in units
        of the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified in units
        of the tip radius.
    radius : float
        Radius of curvature of the AFM tip.
    semi_maj_axis : float
        Semi-major axis length of the effective spheroid from the finite
        dipole model.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `harmonic`.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference_[1]. The function
    :math:`\alpha_{eff}` is implemented here as
    `finite_dipole.bulk.eff_pol_0`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta is None:
            raise ValueError("Either `eps_sample` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_environment, eps_sample)
        else:
            warnings.warn("`beta` overrides `eps_sample` when both are specified.")

    if x_0 is None:
        x_0 = 1.31 * semi_maj_axis / (semi_maj_axis + 2 * radius)

    # Set oscillation centre  so AFM tip touches sample at z = 0
    z_0 = z + tapping_amplitude + radius

    alpha_eff = demod(
        eff_pol_0,
        z_0,
        tapping_amplitude,
        harmonic,
        f_args=(beta, x_0, x_1, radius, semi_maj_axis, g_factor),
    )

    return alpha_eff
