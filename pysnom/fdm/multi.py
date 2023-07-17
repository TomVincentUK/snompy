"""
Multilayer finite dipole model (:mod:`pysnom.fdm.multi`)
========================================================

.. currentmodule:: pysnom.fdm.multi

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the multilayer
finite dipole model (FDM).

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n
    eff_pol
    geom_func

"""
import numpy as np

from .._defaults import defaults
from ..demodulate import demod


def geom_func(z_tip, d_image, r_tip, L_tip, g_factor):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for the multilayer finite dipole
    model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    d_image : float
        Depth of an image charge induced below the upper surface of a stack
        of interfaces.
    r_tip : float
        Radius of curvature of the AFM tip.
    L_tip : float
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

    See also
    --------
    pysnom.fdm.bulk.geom_func : The bulk equivalent of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        f =
        \left(
            g - \frac{r_{tip} + z_{tip} + d_{image}}{2 L_{tip}}
        \right)
        \frac{\ln{\left(\frac{4 L_{tip}}{r_{tip} + 2 z_{tip} + 2 d_{image}}\right)}}
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`d_{image}` is `d_image`, :math:`r_{tip}` is
    `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g` is `g_factor`.
    This is given as equation (11) in reference [1]_.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    return (
        (g_factor - (r_tip + z_tip + d_image) / (2 * L_tip))
        * np.log(4 * L_tip / (r_tip + 2 * z_tip + 2 * d_image))
        / np.log(4 * L_tip / r_tip)
    )


def eff_pol(
    z_tip,
    sample,
    r_tip=None,
    L_tip=None,
    g_factor=None,
    d_Q0=None,
    d_Q1=None,
    n_lag=None,
):
    r"""Return the effective probe-sample polarizability using the
    multilayer finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    sample : :class:`pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate.
    d_Q0 : float
        Depth of an induced charge 0 within the tip. Specified in units of
        the tip radius.
    d_Q1 : float
        Depth of an induced charge 1 within the tip. Specified in units of
        the tip radius.
    r_tip : float
        Radius of curvature of the AFM tip.
    L_tip : float
        Semi-major axis length of the effective spheroid from the finite
        dipole model.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.
    n_lag : int
        The order of the Laguerre polynomial used by
        :func:`surf_pot_and_field`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    pysnom.fdm.bulk.eff_pol : The bulk equivalent of this function.
    eff_pol_n : The modulated/demodulated version of this function.
    geom_func : Multilayer geometry function.
    surf_pot_and_field : Surface electric potential and field.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} =
        1
        + \frac{\beta_{image, 0} f_{geom, ML}(z_{tip}, d_{image, 0}, r_{tip}, L_{tip}, g)}
        {2 (1 - \beta_{image, 1} f_{geom, ML}(z_{tip}, d_{image, 1}, r_{tip}, L_{tip}, g))}

    where :math:`\alpha_{eff}` is `\alpha_eff`; :math:`\beta_{image, i}`
    and :math:`d_{image, i}` are the relative charge and depth of an image
    charge induced by a charge in the tip at :math:`d_{Qi}`
    (:math:`i=0, 1`), given by `d_Q0` and `d_Q1`; :math:`r_{tip}` is `r_tip`,
    :math:`L_{tip}` is `L_tip`, :math:`g` is `g_factor`, and
    :math:`f_{geom, ML}` is a function encapsulating the geometric
    properties of the tip-sample system for the multilayer finite dipole
    model. This is a modified version of equation (3) from reference [1]_.
    The function :math:`f_{geom, ML}` is implemented here as
    :func:`geom_func`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1 = defaults._fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1
    )

    z_q_0 = z_tip + r_tip * d_Q0
    z_im_0, beta_im_0 = sample.image_depth_and_charge(z_q_0, n_lag)
    f_0 = geom_func(z_tip, z_im_0, r_tip, L_tip, g_factor)

    z_q_1 = z_tip + r_tip * d_Q1
    z_im_1, beta_im_1 = sample.image_depth_and_charge(z_q_1, n_lag)
    f_1 = geom_func(z_tip, z_im_1, r_tip, L_tip, g_factor)

    return 1 + (beta_im_0 * f_0) / (2 * (1 - beta_im_1 * f_1))


def eff_pol_n(
    z_tip,
    A_tip,
    n,
    sample,
    r_tip=None,
    L_tip=None,
    g_factor=None,
    d_Q0=None,
    d_Q1=None,
    n_lag=None,
    n_trapz=None,
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the multilayer finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    sample : :class:`pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate.
    r_tip : float
        Radius of curvature of the AFM tip.
    L_tip : float
        Semi-major axis length of the effective spheroid from the finite
        dipole model.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.
    d_Q0 : float
        Depth of an induced charge 0 within the tip. Specified in units of
        the tip radius.
    d_Q1 : float
        Depth of an induced charge 1 within the tip. Specified in units of
        the tip radius.
    n_lag : complex
        The order of the Laguerre polynomial used by :func:`surf_pot_and_field`.
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.

    See also
    --------
    pysnom.fdm.bulk.eff_pol_n : The bulk equivalent of this function.
    eff_pol : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff = demod(
        eff_pol,
        z_0,
        A_tip,
        n,
        f_args=(sample, r_tip, L_tip, g_factor, d_Q0, d_Q1, n_lag),
        n_trapz=n_trapz,
    )

    return alpha_eff
