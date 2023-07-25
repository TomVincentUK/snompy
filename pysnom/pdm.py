"""
Point dipole model (:mod:`pysnom.pdm`)
======================================

.. currentmodule:: pysnom.pdm

This module provides functions for simulating the results of scanning
near-field optical microscopy (SNOM) experiments by calculating the
effective polarisability using the point dipole model (PDM).

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
    eps_sphere=None,
    alpha_sphere=None,
    n_trapz=None,
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the bulk point dipole model.

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
        and superstrate. Sample must have only one interface for bulk
        methods.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_sphere : complex
        Dielectric function of the sample. Used to calculate
        `alpha_sphere`, and ignored if `alpha_sphere` is specified. If both
        `eps_sphere` and `alpha_sphere` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_sphere : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.
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
    pysnom.fdm.eff_pol_n :
        The finite dipole model equivalent of this function.
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

    If `eps_sphere` is specified it is used to calculate `alpha_sphere`
    according to

    .. math ::

        \alpha_{t} = 4 \pi r_{tip}^3 \frac{\varepsilon_t - 1}{\varepsilon_t + 2}

    where :math:`\alpha_{t}` is `alpha_sphere`, :math:`r_{tip}` is `r_tip` and
    :math:`\varepsilon_t` is `eps_t`, which is given as equation (3.1) in
    reference [2]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.
    .. [2] F. Keilmann and R. Hillenbrand, “Near-field microscopy by
       elastic light scattering from a tip,” Philos. Trans. R. Soc. London.
       Ser. A Math. Phys. Eng. Sci., vol. 362, no. 1817, pp. 787–805, Apr.
       2004, doi: 10.1098/rsta.2003.1347.

    """
    # Set oscillation centre  so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff = demod(
        eff_pol,
        z_0,
        A_tip,
        n,
        f_args=(sample, r_tip, eps_sphere, alpha_sphere),
        n_trapz=n_trapz,
    )

    return alpha_eff


def eff_pol(z_tip, sample, r_tip=None, eps_sphere=None, alpha_sphere=None):
    r"""Return the effective probe-sample polarizability using the bulk
    point dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    sample : :class:`pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_sphere : complex
        Dielectric function of the sample. Used to calculate
        `alpha_sphere`, and ignored if `alpha_sphere` is specified. If both
        `eps_sphere` and `alpha_sphere` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_sphere : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    pysnom.fdm.eff_pol :
        The finite dipole model (FDM) equivalent of this function.
    eff_pol_n : The modulated/demodulated version of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} = \frac{\alpha_t}{1 - f_{geom} \beta}

    where :math:`\alpha_{eff}` is `alpha_eff`, :math:`\alpha_{t}` is
    `alpha_sphere`, :math:`\beta` is `beta`, and :math:`f_{geom}` is a
    function encapsulating various geometric properties of the tip-sample
    system, implemented here as :func:`pysnom.pdm.geom_func`.
    This is given as equation (14) in reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.

    """
    # Set defaults
    r_tip, alpha_sphere = defaults._pdm_defaults(r_tip, eps_sphere, alpha_sphere)

    beta = sample.refl_coef_qs()
    f_geom = geom_func(z_tip, r_tip, alpha_sphere)

    return alpha_sphere / (1 - f_geom * beta)


def geom_func(z_tip, r_tip, alpha_sphere):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for bulk point dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    r_tip : float
        Radius of curvature of the AFM tip.
    alpha_sphere : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    f_geom : complex
        A complex number encapsulating geometric properties of the tip-
        sample system.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{geom} = \frac{\alpha_{sphere}}{16 \pi (r_{tip} + z_{tip})^3}

    where :math:`z_{tip}` is `z_tip`, :math:`r_{tip}` is `r_tip`, and
    :math:`\alpha_{sphere}` is `alpha_sphere`. This is adapted from
    equation (14) of reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.
    """
    return alpha_sphere / (16 * np.pi * (r_tip + z_tip) ** 3)


def taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_sphere, n_trapz):
    r"""Return the coefficient for the power of reflection coefficient used
    by the Taylor series representation of the bulk PDM.

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
    alpha_sphere : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.
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

        a_j = \hat{F_n}[f_{geom}^j],

    where :math:`\hat{F_n}[f_{geom}(j)]` is the :math:`n^{th}` Fourier
    coefficient of the function :math:`f_{geom}^j`, and :math:`f_{geom}`
    is implemented here as :func:`geom_func_taylor`.

    """
    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    a_j = demod(
        lambda z, j_taylor, r_tip, alpha_sphere: geom_func(z, r_tip, alpha_sphere)
        ** j_taylor,
        z_0,
        A_tip,
        n,
        f_args=(j_taylor, r_tip, alpha_sphere),
        n_trapz=n_trapz,
    )
    return a_j


def eff_pol_n_taylor(
    z_tip,
    A_tip,
    n,
    sample,
    r_tip=None,
    eps_sphere=None,
    alpha_sphere=None,
    n_trapz=None,
    n_tayl=None,
):
    r"""Return the effective probe-sample polarizability using the point
    dipole model, demodulated at harmonics of the tapping frequency, using
    a Taylor series representation of the bulk PDM.

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
        and superstrate. Sample must have only one interface for bulk
        methods.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_sphere : complex
        Dielectric function of the sample. Used to calculate
        `alpha_sphere`, and ignored if `alpha_sphere` is specified. If both
        `eps_sphere` and `alpha_sphere` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_sphere : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.
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
    :math:`\alpha_{eff, n} = \sum_{j=0}^{J} a_j \beta^j`, where
    :math:`\beta` is `beta`, :math:`j` is the index of the Taylor series,
    :math:`J` is `n_tayl` and :math:`a_j` is the Taylor coefficient,
    implemented here as :func:`taylor_coef`.
    """
    # Set defaults
    r_tip, alpha_sphere = defaults._pdm_defaults(r_tip, eps_sphere, alpha_sphere)
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl

    beta = sample.refl_coef_qs()

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, beta, r_tip, alpha_sphere)
    )

    coefs = taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_sphere, n_trapz)
    alpha_eff = np.sum(coefs * beta**j_taylor, axis=0)
    return alpha_eff


def refl_coef_qs_from_eff_pol_n(
    z_tip,
    A_tip,
    n,
    alpha_eff_n,
    r_tip=None,
    eps_sphere=None,
    alpha_sphere=None,
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
    r_tip, alpha_sphere = defaults._pdm_defaults(r_tip, eps_sphere, alpha_sphere)
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl
    beta_threshold = (
        defaults.beta_threshold if beta_threshold is None else beta_threshold
    )

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, alpha_eff_n, r_tip, alpha_sphere)
    )
    coefs = taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_sphere, n_trapz)

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
    z_tip, alpha_eff, r_tip=None, eps_sphere=None, alpha_sphere=None
):
    r"""Return the quasistatic reflection coefficient corresponding to a
    particular effective polarizability using the point dipole model.

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
    r_tip, alpha_sphere = defaults._pdm_defaults(r_tip, eps_sphere, alpha_sphere)

    f_geom = geom_func(z_tip, r_tip, alpha_sphere)

    beta = (alpha_eff - alpha_sphere) / (f_geom * alpha_eff)

    return beta
