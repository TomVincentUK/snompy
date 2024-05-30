"""
Point dipole model (:mod:`snompy.pdm`)
======================================

.. currentmodule:: snompy.pdm

This module provides functions for simulating the results of scanning
near-field optical microscopy (SNOM) experiments by calculating the
effective polarizability using the point dipole model (PDM).

Standard functions
^^^^^^^^^^^^^^^^^^
Functions for the effective polarizability of an AFM tip coupled to a
sample, and the effective polarizability demodulated at higher harmonics.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n
    eff_pol

Inverse functions
^^^^^^^^^^^^^^^^^
Functions to return the quasistatic reflection coefficient of a sample
based on the effective polarizability of an AFM tip coupled to the sample.

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
from .sample import permitivitty


def eff_pol_n(sample, A_tip, n, z_tip=None, n_trapz=None, **kwargs):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the bulk point dipole model.

    Parameters
    ----------
    sample : :class:`snompy.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    z_tip : float
        Height of the tip above the sample.
    n_trapz : int
        The number of intervals used by :func:`snompy.demodulate.demod` for
        the trapezium-method integration.
    **kwargs : dict, optional
        Extra keyword arguments are passed to :func:`eff_pol`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.

    See also
    --------
    snompy.fdm.eff_pol_n :
        The finite dipole model equivalent of this function.
    eff_pol : The unmodulated/demodulated version of this function.
    snompy.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol`.

    If `eps_tip` is specified it is used to calculate `alpha_tip`
    according to

    .. math ::

        \alpha_{tip} = 4 \pi r_{tip}^3
        \frac{\varepsilon_{tip} - 1}{\varepsilon_{tip} + 2}

    where :math:`\alpha_{tip}` is `alpha_tip`, :math:`r_{tip}` is
    `r_tip` and :math:`\varepsilon_{tip}` is `eps_tip`, which is given as
    equation (3.1) in reference [2]_.

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
    # Set defaults
    z_tip = defaults.z_tip if z_tip is None else z_tip

    # Set oscillation centre  so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff_n = demod(
        lambda x, **kwargs: eff_pol(z_tip=x, **kwargs),
        z_0,
        A_tip,
        n,
        n_trapz=n_trapz,
        sample=sample,
        **kwargs
    )

    return alpha_eff_n


def eff_pol(sample, z_tip=None, r_tip=None, eps_tip=None, alpha_tip=None):
    r"""Return the effective probe-sample polarizability using the bulk
    point dipole model.

    Parameters
    ----------
    sample : :class:`snompy.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
    z_tip : float
        Height of the tip above the sample.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_tip : complex
        Dielectric function of the tip. Used to calculate `alpha_tip`, and
        ignored if `alpha_tip` is specified. If both `eps_tip` and
        `alpha_tip` are None, the model sphere is assumed to be perfectly
        conducting.
    alpha_tip : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    snompy.fdm.eff_pol :
        The finite dipole model (FDM) equivalent of this function.
    eff_pol_n : The modulated/demodulated version of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} = \frac{\alpha_{tip}}{1 - f \beta}

    where :math:`\alpha_{eff}` is `alpha_eff`, :math:`\alpha_{tip}` is
    `alpha_tip`, :math:`\beta` is `beta`, and :math:`f` is a
    function encapsulating various geometric properties of the tip-sample
    system, implemented here as :func:`snompy.pdm.geom_func`.
    This is given as equation (14) in reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.

    """
    # Set defaults
    z_tip = defaults.z_tip if z_tip is None else z_tip
    r_tip, alpha_tip = defaults._pdm_defaults(r_tip, eps_tip, alpha_tip)

    beta = sample.refl_coef_qs()
    f_geom = geom_func(z_tip, r_tip, alpha_tip)

    return alpha_tip / (1 - f_geom * beta)


def geom_func(z_tip, r_tip, alpha_tip):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for bulk point dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    r_tip : float
        Radius of curvature of the AFM tip.
    alpha_tip : complex
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

        f = \frac{\alpha_{tip}}{16 \pi (r_{tip} + z_{tip})^3}

    where :math:`z_{tip}` is `z_tip`, :math:`r_{tip}` is `r_tip`, and
    :math:`\alpha_{tip}` is `alpha_tip`. This is adapted from
    equation (14) of reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.
    """
    return alpha_tip / (16 * np.pi * (r_tip + z_tip) ** 3)


def taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_tip, n_trapz):
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
    alpha_tip : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.
    n_trapz : int
        The number of intervals used by :func:`snompy.demodulate.demod` for
        the trapezium-method integration.

    Returns
    -------
    a_j : complex
        Coefficient for the power of reflection coefficient used by the
        Taylor series representation of the bulk FDM

    See also
    --------
    snompy.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements

    .. math::

        a_j = \hat{F_n}[f^j],

    where :math:`\hat{F_n}[f(j)]` is the :math:`n^{th}` Fourier
    coefficient of the function :math:`f^j`, and :math:`f`
    is implemented here as :func:`geom_func`.

    """
    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    a_j = demod(
        lambda z, j_taylor, r_tip, alpha_tip: geom_func(z, r_tip, alpha_tip)
        ** j_taylor,
        z_0,
        A_tip,
        n,
        f_args=(j_taylor, r_tip, alpha_tip),
        n_trapz=n_trapz,
    )
    return a_j


def eff_pol_n_taylor(
    sample,
    A_tip,
    n,
    z_tip=None,
    r_tip=None,
    eps_tip=None,
    alpha_tip=None,
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
    sample : :class:`snompy.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    z_tip : float
        Height of the tip above the sample.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_tip : complex
        Dielectric function of the sample. Used to calculate
        `alpha_tip`, and ignored if `alpha_tip` is specified. If both
        `eps_tip` and `alpha_tip` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_tip : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.
    n_trapz : int
        The number of intervals used by :func:`snompy.demodulate.demod` for
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
    snompy.demodulate.demod :
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
    z_tip = defaults.z_tip if z_tip is None else z_tip
    r_tip, alpha_tip = defaults._pdm_defaults(r_tip, eps_tip, alpha_tip)
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl

    beta = sample.refl_coef_qs()

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, beta, r_tip, alpha_tip)
    )

    coefs = taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_tip, n_trapz)
    alpha_eff = np.sum(coefs * beta**j_taylor, axis=0)
    return alpha_eff


def refl_coef_qs_from_eff_pol_n(
    alpha_eff_n,
    A_tip,
    n,
    z_tip=None,
    r_tip=None,
    eps_tip=None,
    alpha_tip=None,
    n_trapz=None,
    n_tayl=None,
    beta_threshold=None,
    reject_negative_eps_imag=False,
    reject_subvacuum_eps_abs=False,
    eps_env=None,
):
    r"""Return the quasistatic reflection coefficient corresponding to a
    particular effective polarizability, demodulated at higher harmonics,
    using a Taylor series representation of the FDM.

    Parameters
    ----------
    alpha_eff_n : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    z_tip : float
        Height of the tip above the sample.
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
        The number of intervals used by :func:`snompy.demodulate.demod` for
        the trapezium-method integration.
    n_tayl : int
        Maximum power index for the Taylor series in `beta`.
    beta_threshold : float
        The maximum amplitude of returned `beta` values determined to be
        valid.
    reject_negative_eps_imag : bool
        True if values of `beta` corresponding to bulk samples with
        negative imaginary permitivitty should be rejected as invalid
        results.
    reject_subvacuum_eps_abs : bool
        True if values of `beta` corresponding to bulk samples with
        permitivitty whose magnitude is less than the vacuum permitivitty
        (1.0) should be rejected as invalid results.
    eps_env : array_like
        Permitivitty of the environment. Used to calculate the sample
        permitivitty when `reject_negative_eps_imag` or
        `reject_subvacuum_eps_abs` are True.

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
    z_tip = defaults.z_tip if z_tip is None else z_tip
    r_tip, alpha_tip = defaults._pdm_defaults(r_tip, eps_tip, alpha_tip)
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl
    beta_threshold = (
        defaults.beta_threshold if beta_threshold is None else beta_threshold
    )
    eps_env = defaults.eps_env if eps_env is None else eps_env

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, alpha_eff_n, r_tip, alpha_tip)
    )
    coefs = taylor_coef(z_tip, j_taylor, A_tip, n, r_tip, alpha_tip, n_trapz)

    offset_coefs = np.where(j_taylor == 0, coefs - alpha_eff_n, coefs)
    all_roots = np.apply_along_axis(lambda c: Polynomial(c).roots(), 0, offset_coefs)

    # Identify valid solutions
    valid = np.abs(all_roots) <= beta_threshold

    eps = permitivitty(all_roots, eps_env)
    if reject_negative_eps_imag:
        valid &= eps.imag >= 0
    if reject_subvacuum_eps_abs:
        valid &= np.abs(eps) >= 1

    # Sort roots by validity
    all_roots = np.take_along_axis(all_roots, valid.argsort(axis=0), axis=0)
    valid = np.take_along_axis(valid, valid.argsort(axis=0), axis=0)

    # Different numbers of solutions may be returned for different inputs
    # Here we remove any slices along the first axis that have no valid solutions
    slice_contains_valid = valid.any(axis=tuple(range(1, np.ndim(all_roots))))
    unmasked_roots = all_roots[slice_contains_valid]
    valid = valid[slice_contains_valid]

    # Sort remaining roots by abs value
    valid = np.take_along_axis(valid, np.abs(unmasked_roots).argsort(axis=0), axis=0)
    unmasked_roots = np.take_along_axis(
        unmasked_roots, np.abs(unmasked_roots).argsort(axis=0), axis=0
    )

    # Masked array with any remaining invalid results hidden by the mask
    beta = np.ma.array(unmasked_roots, mask=~valid)
    return beta


def refl_coef_qs_from_eff_pol(
    alpha_eff, z_tip=None, r_tip=None, eps_tip=None, alpha_tip=None
):
    r"""Return the quasistatic reflection coefficient corresponding to a
    particular effective polarizability using the point dipole model.

    Parameters
    ----------
    alpha_eff : complex
        Effective polarizability of the tip and sample.
    z_tip : float
        Height of the tip above the sample.
    r_tip : float
        Radius of curvature of the AFM tip.
    eps_tip : complex
        Dielectric function of the sample. Used to calculate
        `alpha_tip`, and ignored if `alpha_tip` is specified. If both
        `eps_tip` and `alpha_tip` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_tip : complex
        Polarizability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    beta : complex, masked array
        Quasistatic  reflection coefficient of the interface.

    See also
    --------
    eff_pol : The inverse of this function.
    refl_coef_qs_from_eff_pol_n : The demodulated equivalent of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        \beta = \frac{(\alpha_{eff} - \alpha_{tip})}{f \alpha_{eff}}

    where :math:`\alpha_{eff}` is `alpha_eff`, and :math:`f` is a function
    encapsulating the PDM geometry, taken from reference [1]_.
    Here it is given by :func:`geom_func`.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.

    """
    # Set defaults
    z_tip = defaults.z_tip if z_tip is None else z_tip
    r_tip, alpha_tip = defaults._pdm_defaults(r_tip, eps_tip, alpha_tip)

    f_geom = geom_func(z_tip, r_tip, alpha_tip)

    beta = (alpha_eff - alpha_tip) / (f_geom * alpha_eff)

    return beta
