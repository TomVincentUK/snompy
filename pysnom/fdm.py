"""
Finite dipole model (:mod:`pysnom.fdm`)
=======================================

.. currentmodule:: pysnom.fdm

This module provides functions for simulating the results of scanning
near-field optical microscopy (SNOM) experiments by calculating the
effective polarisability using the finite dipole model (FDM).

Standard functions
^^^^^^^^^^^^^^^^^^
Functions for the effective polarisability of an AFM tip coupled to a
sample, and the effective polarisability demodulated at higher harmonics.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n
    eff_pol

Inverse functions
^^^^^^^^^^^^^^^^^
Functions to return the quasistatic reflection coefficient of a sample
based on the effective polarisability of an AFM tip coupled to the sample.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    refl_coef_qs_from_eff_pol_n
    refl_coef_qs_from_eff_pol


Internal functions
^^^^^^^^^^^^^^^^^^
These functions are used by the standard functions in this module. In
most cases you shouldn't need to call these functions directly.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    geom_func
    geom_func_multi
    geom_func_taylor
    taylor_coef
    eff_pol_n_taylor

"""

import numpy as np
from numpy.polynomial import Polynomial

from ._defaults import defaults
from ._utils import _pad_for_broadcasting
from .demodulate import demod


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
    d_Qa=None,
    n_lag=None,
    method=None,
    n_trapz=None,
):
    r"""Return the effective probe-sample polarizability using the finite
    dipole model, demodulated at harmonics of the tapping frequency.

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
    d_Qa : float
        Depth of a single representative charge within the tip. Specified
        in units of the tip radius. Used by the Mester implementation of
        the finite dipole model to calculate the effective quasistatic
        reflection coefficient for the tip.
    n_lag : int
        The order of the Gauss-Laguerre integration used by the "Hauer" and
        "Mester" methods.
    method : {"bulk", "Hauer", "Mester"}
        The method of the finite dipole model to use. See :func:`eff_pol`
        for descriptions of the different methods.
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.

    Returns
    -------
    alpha_eff_n : complex
        Demodulated effective polarizability of the tip and sample.

    See also
    --------
    eff_pol : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    """
    # Set oscillation centre so AFM tip just touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff_n = demod(
        eff_pol,
        z_0,
        A_tip,
        n,
        f_args=(sample, r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa, n_lag, method),
        n_trapz=n_trapz,
    )

    return alpha_eff_n


def eff_pol(
    z_tip,
    sample,
    r_tip=None,
    L_tip=None,
    g_factor=None,
    d_Q0=None,
    d_Q1=None,
    d_Qa=None,
    n_lag=None,
    method=None,
):
    r"""Return the effective probe-sample polarizability using the finite
    dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
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
    d_Qa : float
        Depth of a single representative charge within the tip. Specified
        in units of the tip radius. Used by the Mester implementation of
        the finite dipole model to calculate the effective quasistatic
        reflection coefficient for the tip.
    n_lag : int
        The order of the Gauss-Laguerre integration used by the "Hauer" and
        "Mester" methods.
    method : {"bulk", "Hauer", "Mester"}
        The method of the finite dipole model to use. See Notes for the
        description of each method. Defaults to "bulk" for bulk samples and
        "Hauer" otherwise.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    eff_pol_n :
        This function demodulated at chosen harmonics of the tapping
        frequency.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} = 1 + \frac
            {\beta_0 f_{geom, 0}(z_{tip}, r_{tip}, L_{tip}, g)}
            {2 (1 - \beta_1 f_{geom, 1}(z_{tip}, r_{tip}, L_{tip}, g))}

    where :math:`\alpha_{eff}` is `\alpha_eff`, :math:`z_{tip}` is `z_tip`,
    :math:`r_{tip}` is `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g`
    is `g_factor`.

    The definitions of :math:`\beta_i` and :math:`f_{geom, i}` depend on
    the FDM method used, and are described below.

    Method "bulk" is the bulk Hauer method given in reference [1]_. Here,
    :math:`\beta_0 = \beta_1 = \beta`, the momentum independent quasistatic
    reflection coefficient of the sample, which is calculated from
    :func:`pysnom.sample.Sample.refl_coef_qs` (with argument `q = 0`).
    :math:`f_{geom, i}` is given by :func:`geom_func`, with arguments
    `(z_tip, d_Qi, r_tip, L_tip, g_factor)` where `d_Qi` is given by
    `d_Q0`, `d_Q1` for :math:`i = 0, 1`.

    Method "Hauer" is the multilayer Hauer method given in reference [1]_.
    Here, :math:`\beta_i`, is the relative charge of an image of charge
    :math:`Q_i` below the sample at depth :math:`z_{Q'i}` below the
    surface. :math:`\beta_i` and :math:`d_{z'i}` are calculated from
    :func:`pysnom.sample.Sample.image_depth_and_charge`.
    :math:`f_{geom, i}` is given by :func:`geom_func_multi`, with arguments
    `(z_tip, z_im_i, r_tip, L_tip, g_factor)` where `z_im_i` is given by
    :math:`d_{z'i}` for :math:`i = 0, 1`.

    Method "Mester" is described in reference [2]_. Here
    :math:`\beta_0 = \beta_1 = \overline{\beta}`, the effective reflection
    coefficient for a test charge :math:`Q_a`, evaluated at the position of
    the charge itself, which is calculated from
    :func:`pysnom.sample.Sample.refl_coef_qs_above_surf`. The definition of
    :math:`f_{geom, i}` is the same as for the bulk Hauer method.


    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
           model for scattering infrared near-field microscopy on layered
           systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
           doi: 10.1364/OE.20.013173.
    .. [2] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
           R. Hillenbrand, “Subsurface chemical nanoidentification by
           nano-FTIR spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359,
           Dec. 2020, doi: 10.1038/s41467-020-17034-6.
    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa = defaults._fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa
    )

    # Default to one of the Hauer methods based on sample type.
    if method is None:
        method = "Hauer" if sample.multilayer else "bulk"

    if method == "bulk":
        if sample.multilayer:
            raise ValueError("`method`='bulk' cannot be used for multilayer samples.")
        beta_0 = beta_1 = sample.refl_coef_qs()

        f_0 = geom_func(z_tip, d_Q0, r_tip, L_tip, g_factor)
        f_1 = geom_func(z_tip, d_Q1, r_tip, L_tip, g_factor)
    elif method == "Hauer":
        z_Q0 = z_tip + r_tip * d_Q0
        z_Q1 = z_tip + r_tip * d_Q1
        z_im0, beta_0 = sample.image_depth_and_charge(z_Q0, n_lag)
        z_im1, beta_1 = sample.image_depth_and_charge(z_Q1, n_lag)

        f_0 = geom_func_multi(z_tip, z_im0, r_tip, L_tip, g_factor)
        f_1 = geom_func_multi(z_tip, z_im1, r_tip, L_tip, g_factor)
    elif method == "Mester":
        z_Qa = z_tip + r_tip * d_Qa
        beta_0 = beta_1 = sample.refl_coef_qs_above_surf(z_Qa, n_lag)

        f_0 = geom_func(z_tip, d_Q0, r_tip, L_tip, g_factor)
        f_1 = geom_func(z_tip, d_Q1, r_tip, L_tip, g_factor)
    else:
        raise ValueError("`method` must be one of `bulk`, `Hauer`, or `Mester`.")

    alpha_eff = 1 + (beta_0 * f_0) / (2 * (1 - beta_1 * f_1))

    return alpha_eff


def geom_func(z_tip, d_Q, r_tip, L_tip, g_factor):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for bulk finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    d_Q : float
        Depth of an induced charge within the tip. Specified in units of
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

    Returns
    -------
    f_n : complex
        A complex number encapsulating geometric properties of the tip-
        sample system.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{geom, i} =
        \left(
            g - \frac{r_{tip} + 2 z_{tip} + r_{tip} d_{Qi}}{2 L_{tip}}
        \right)
        \frac{\ln{\left(
            \frac{4 L_{tip}}{r_{tip} + 4 z_{tip} + 2 r_{tip} d_{Qi}}
        \right)}}
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`d_{Qi}` is `d_Q`,
    :math:`r_{tip}` is `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g`
    is `g_factor`. This is given as equation (2) in reference [1]_.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    return (
        (g_factor - (r_tip + 2 * z_tip + d_Q * r_tip) / (2 * L_tip))
        * np.log(4 * L_tip / (r_tip + 4 * z_tip + 2 * d_Q * r_tip))
        / np.log(4 * L_tip / r_tip)
    )


def geom_func_multi(z_tip, d_image, r_tip, L_tip, g_factor):
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

    Notes
    -----
    This function implements the equation

    .. math::

        f_{geom, i} =
        \left(
            g - \frac{r_{tip} + z_{tip} + z_{Q'i}}{2 L_{tip}}
        \right)
        \frac{\ln{\left(
            \frac{4 L_{tip}}{r_{tip} + 2 z_{tip} + 2 z_{Q'i}}
        \right)}
        }
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`z_{Q'i}` is `d_image`,
    :math:`r_{tip}` is `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g`
    is `g_factor`. This is given as equation (11) in reference [1]_.

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


def geom_func_taylor(z_tip, j_taylor, r_tip, L_tip, g_factor, d_Q0, d_Q1):
    r"""The height-dependent part of the separable Taylor series expression
    for the FDM.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    j_taylor : integer
        The corresponding power of the reflection coefficient in the Taylor
        series.
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

    Returns
    -------
    f_t : complex
        The height-dependent part of the separable taylor series expression
        for the bulk FDM.

    See also
    --------
    taylor_coef :
        Function that uses this to calculate the Taylor series coefficients
        for the bulk FDM.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{t} = f_{geom, 0} f_{geom, 1}^{j-1}

    where :math:`f_{t}` is `f_t`, :math:`j` is `j_taylor`,
    and :math:`f_{geom, i}` is a function encapsulating the geometric
    properties of the tip-sample system, implemented here as
    :func:`geom_func`.

    """
    f_0 = geom_func(z_tip, d_Q0, r_tip, L_tip, g_factor)
    f_1 = geom_func(z_tip, d_Q1, r_tip, L_tip, g_factor)
    return f_0 * f_1 ** (j_taylor - 1)


def taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, L_tip, g_factor, d_Q0, d_Q1, n_trapz):
    r"""Return the coefficient for the power of reflection coefficient used
    by the Taylor series representation of the bulk FDM.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    j_taylor : integer
        The corresponding power of the reflection coefficient in the Taylor
        series.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
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
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.

    Returns
    -------
    a_j : complex
        Coefficient for the power of reflection coefficient used by the
        Taylor series representation of the bulk FDM

    See also
    --------
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements

    .. math::

        a_j = \frac{1}{2} \hat{F_n}[f_t(j)],

    where :math:`\hat{F_n}[f_t(j)]` is the :math:`n^{th}` Fourier
    coefficient of the function :math:`f_t(j)`, which is implemented here
    as :func:`geom_func_taylor`.
    """
    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    non_zero_terms = (
        demod(
            geom_func_taylor,
            z_0,
            A_tip,
            n,
            f_args=(j_taylor, r_tip, L_tip, g_factor, d_Q0, d_Q1),
            n_trapz=n_trapz,
        )
        / 2
    )
    return np.where(j_taylor == 0, np.where(n == 0, 1, 0), non_zero_terms)


def eff_pol_n_taylor(
    z_tip,
    A_tip,
    n,
    sample,
    r_tip=None,
    L_tip=None,
    g_factor=None,
    d_Q0=None,
    d_Q1=None,
    d_Qa=None,
    n_lag=None,
    method=None,
    n_trapz=None,
    n_tayl=None,
):
    r"""Return the effective probe-sample polarizability using the finite
    dipole model, demodulated at harmonics of the tapping frequency, using
    a Taylor series representation of the bulk FDM.

    .. note::
        This function primarily exists to check the validity of the Taylor
        approximation to `eff_pol_n` which is used by other functions. For
        best performance it is recommended to use `eff_pol_n`.

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
    d_Qa : float
        Depth of a single representative charge within the tip. Specified
        in units of the tip radius. Used by the Mester implementation of
        the finite dipole model to calculate the effective quasistatic
        reflection coefficient for the tip.
    n_lag : int
        The order of the Gauss-Laguerre integration used by the "Hauer" and
        "Mester" methods.
    method : {"bulk", "Mester"}
        The method of the finite dipole model to use. See :func:`eff_pol`
        for descriptions of the different methods.
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.
    n_tayl : int
        Maximum power index for the Taylor series in `beta`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.

    See also
    --------
    eff_pol_n : The non-Taylor series version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function is valid only for reflection coefficients, `beta`, with
    magnitudes less than around 1. For a more generally applicable function
    use :func:`eff_pol_n`

    This function implements
    :math:`\alpha_{eff, n} = \delta(n) + \sum_{j=1}^{J} a_j \beta^j`, where
    :math:`\delta` is the Dirac delta function :math:`\beta` is `beta`,
    :math:`j` is the index of the Taylor series, :math:`J` is
    `n_tayl` and :math:`a_j` is the Taylor coefficient, implemented
    here as :func:`taylor_coef`.
    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa = defaults._fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa
    )
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl

    # Default to one of the Hauer methods based on sample type.
    if method is None:
        method = "Hauer" if sample.multilayer else "bulk"

    if method == "bulk":
        if sample.multilayer:
            raise ValueError("`method`='bulk' cannot be used for multilayer samples.")
        beta = sample.refl_coef_qs()
    elif method == "Mester":
        z_Qa = z_tip + r_tip * d_Qa
        beta = sample.refl_coef_qs_above_surf(z_Qa, n_lag)
    else:
        raise ValueError("`method` must be one of `bulk`, or `Mester`.")

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, beta, r_tip, L_tip, g_factor, d_Q0, d_Q1)
    )

    coefs = taylor_coef(
        z_tip, j_taylor, A_tip, n, r_tip, L_tip, g_factor, d_Q0, d_Q1, n_trapz
    )
    alpha_eff = np.sum(coefs * beta**j_taylor, axis=0)
    return alpha_eff


def refl_coef_qs_from_eff_pol_n(
    z_tip,
    A_tip,
    n,
    alpha_eff_n,
    r_tip=None,
    L_tip=None,
    g_factor=None,
    d_Q0=None,
    d_Q1=None,
    n_trapz=None,
    n_tayl=None,
    beta_threshold=None,
):
    r"""Return the quasistatic reflection coefficient corresponding to a
    particular effective polarizability, demodulated at higher harmonics,
    using a Taylor series representation of the FDM.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    alpha_eff_n : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.
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
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.
    n_tayl : int
        Maximum power index for the Taylor series in `beta`.
    beta_threshold : float
        The maximum amplitude of returned `beta` values determined to be
        valid.

    Returns
    -------
    beta : complex, masked array
        Quasistatic  reflection coefficient of the interface.

    See also
    --------
    taylor_coef :
        Function that calculates the Taylor series coefficients for the
        bulk FDM.
    eff_pol_n_taylor : The inverse of this function.

    Notes
    -----
    This function is valid only `alpha_eff_n` values corresponding to`beta`
    magnitudes less than around 1.

    This function solves, for :math:`\beta`,
    :math:`\alpha_{eff, n} = \delta(n) + \sum_{j=1}^{J} a_j \beta^j`, where
    :math:`\delta` is the Dirac delta function :math:`\beta` is `beta`,
    :math:`j` is the index of the Taylor series, :math:`J` is
    `n_tayl` and :math:`a_j` is the Taylor coefficient, implemented
    here as :func:`taylor_coef`.

    There may be multiple possible solutions (or none) for different
    inputs, so this function returns a masked array with first dimension
    whose length is the maximum number of solutions returned for all input
    values. Values which are invalid are masked.
    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1, _ = defaults._fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa=None
    )
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl
    beta_threshold = (
        defaults.beta_threshold if beta_threshold is None else beta_threshold
    )

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl),
        (z_tip, A_tip, n, alpha_eff_n, r_tip, L_tip, g_factor, d_Q0, d_Q1),
    )
    coefs = taylor_coef(
        z_tip, j_taylor, A_tip, n, r_tip, L_tip, g_factor, d_Q0, d_Q1, n_trapz
    )

    offset_coefs = np.where(j_taylor == 0, coefs - alpha_eff_n, coefs)
    all_roots = np.apply_along_axis(lambda c: Polynomial(c).roots(), 0, offset_coefs)

    # Sort roots by abs value
    all_roots = np.take_along_axis(all_roots, np.abs(all_roots).argsort(axis=0), axis=0)

    # Different numbers of solutions may be returned for different inputs
    # Here we remove any slices along the first axis that have no valid solutions
    slice_contains_valid = (np.abs(all_roots) <= beta_threshold).any(
        axis=tuple(range(1, np.ndim(all_roots)))
    )
    unmasked_roots = all_roots[slice_contains_valid]
    beta = np.ma.array(unmasked_roots, mask=np.abs(unmasked_roots) >= beta_threshold)
    return beta


def refl_coef_qs_from_eff_pol(
    z_tip, alpha_eff, r_tip=None, L_tip=None, g_factor=None, d_Q0=None, d_Q1=None
):
    r"""Return the quasistatic reflection coefficient corresponding to a
    particular effective polarizability.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    alpha_eff : complex
        Effective polarizability of the tip and sample.
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
    n_trapz : int
        The number of intervals used by :func:`pysnom.demodulate.demod` for
        the trapezium-method integration.
    n_tayl : int
        Maximum power index for the Taylor series in `beta`.
    beta_threshold : float
        The maximum amplitude of returned `beta` values determined to be
        valid.

    Returns
    -------
    beta : complex, masked array
        Quasistatic  reflection coefficient of the interface.

    See also
    --------
    eff_pol : The inverse of this function.
    refl_coef_qs_from_n : The demodulated equivalent of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        \beta = \frac
            {2 (\alpha_{eff} - 1)}
            {f_{geom, 0} + 2 f_{geom, 1} (\alpha_{eff} - 1)}

    where :math:`\alpha_{eff}` is `\alpha_eff`, and :math:`f_{geom, i}` is
    a function encapsulating the FDM geometry, taken from reference [1]_.
    Here it is given by :func:`geom_func`, with arguments
    `(z_tip, d_Qi, r_tip, L_tip, g_factor)` where `d_Qi` is replaced by
    `d_Q0`, `d_Q1` for :math:`i = 0, 1`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.

    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1, _ = defaults._fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1, d_Qa=None
    )

    f_0 = geom_func(z_tip, d_Q0, r_tip, L_tip, g_factor)
    f_1 = geom_func(z_tip, d_Q1, r_tip, L_tip, g_factor)

    beta = 2 * (alpha_eff - 1) / (f_0 + 2 * f_1 * (alpha_eff - 1))

    return beta
