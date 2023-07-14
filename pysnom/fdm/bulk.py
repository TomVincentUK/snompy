import numpy as np
from numpy.polynomial import Polynomial

from .. import defaults
from .._utils import _fdm_defaults, _pad_for_broadcasting
from ..demodulate import demod


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

    See also
    --------
    pysnom.fdm.multi.geom_func :
        The multilayer equivalent of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{geom} =
        \left(
            g - \frac{r_{tip} + 2 z_{tip} + r_{tip} d_Q}{2 L_{tip}}
        \right)
        \frac{\ln{\left(\frac{4 L_{tip}}{r_{tip} + 4 z_{tip} + 2 r_{tip} d_Q}\right)}}
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`d_Q` is `d_Q`, :math:`r_{tip}` is
    `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g` is `g_factor`.
    This is given as equation (2) in reference [1]_.

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


def eff_pol(z_tip, sample, r_tip=None, L_tip=None, g_factor=None, d_Q0=None, d_Q1=None):
    r"""Return the effective probe-sample polarizability using the bulk
    finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    sample : `~pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
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
        Depth of an induced charge 1 within the tip. Specified in units
        of the tip radius.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    pysnom.fdm.multi.eff_pol :
        The multilayer equivalent of this function.
    eff_pol_n : The modulated/demodulated version of this function.
    geom_func : Geometry function.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} =
        1
        + \frac{\beta f_{geom}(z_{tip}, d_{Q0}, r_{tip}, L_{tip}, g)}
        {2 (1 - \beta f_{geom}(z_{tip}, d_{Q1}, r_{tip}, L_{tip}, g))}

    where :math:`\alpha_{eff}` is `alpha_eff`, :math:`\beta` is `beta`,
    :math:`z_{tip}` is `z_tip`, :math:`d_{Q0}` is `d_Q0`, :math:`d_{Q1}` is `d_Q1`,
    :math:`r_{tip}` is `r_tip`, :math:`L_{tip}` is `L_tip`, :math:`g` is
    `g_factor`, and :math:`f_{geom}` is a function encapsulating the
    geometric properties of the tip-sample system. This is given as
    equation (3) in reference [1]_. The function :math:`f_{geom}` is
    implemented here as :func:`geom_func`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    # Set defaults
    r_tip, L_tip, g_factor, d_Q0, d_Q1 = _fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1
    )

    beta = sample.refl_coef_qs()

    f_0 = geom_func(z_tip, d_Q0, r_tip, L_tip, g_factor)
    f_1 = geom_func(z_tip, d_Q1, r_tip, L_tip, g_factor)

    return 1 + (beta * f_0) / (2 * (1 - beta * f_1))


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
    n_trapz=None,
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the bulk finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    sample : `~pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
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
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.

    See also
    --------
    pysnom.fdm.multi.eff_pol_n :
        The multilayer equivalent of this function.
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
        f_args=(sample, r_tip, L_tip, g_factor, d_Q0, d_Q1),
        n_trapz=n_trapz,
    )

    return alpha_eff


def geom_func_taylor(z_tip, j_taylor, r_tip, L_tip, g_factor, d_Q0, d_Q1):
    r"""The height-dependent part of the separable Taylor series expression
    for the bulk FDM.

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
    geom_func
    taylor_coef :
        Function that uses this to calculate the Taylor series coefficients
        for the bulk FDM.

    Notes
    -----
    This function implements the equation

    .. math::

        f_{t} = f_{geom}(z_{tip}, d_Q0, r_{tip}, L_{tip}, g) f_{geom}(z_{tip}, d_Q1, r_{tip}, L_{tip}, g)^{j-1}

    where :math:`f_{t}` is `f_t`, :math:`r_{tip}` is `r_tip`, :math:`L_{tip}` is
    `L_tip`, :math:`g` is `g_factor`, :math:`j` is `j_taylor`,
    and :math:`f_{geom}` is a function encapsulating the geometric
    properties of the tip-sample system. This is given as equation (3) in
    reference [1]_. The function :math:`f_{geom}` is implemented here as
    :func:`geom_func`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
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
    geom_func_taylor
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`a_j = \frac{1}{2} \hat{F_n}(f_t)`, where :math:`\hat{F_n}(f_t)`
    is the :math:`n^{th}` Fourier coefficient of the function :math:`f_t`,
    which is implemented here as :func:`geom_func_taylor`.

    This function returns 0 when :math:`j = 0`, because the Taylor series
    representation of the bulk FDM begins at :math:`j = 1`, however
    :class:`numpy.polynomial.polynomial.Polynomial` requires the first index
    to be zero.
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
    return np.where(j_taylor == 0, 0, non_zero_terms)


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
    n_trapz=None,
    n_tayl=None,
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using a Taylor series representation of the bulk FDM.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    sample : `~pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate. Sample must have only one interface for bulk
        methods.
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

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `n`.

    See also
    --------
    taylor_coef :
        Function that calculates the Taylor series coefficients for the
        bulk FDM.
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
    r_tip, L_tip, g_factor, d_Q0, d_Q1 = _fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1
    )
    n_tayl = defaults.n_tayl if n_tayl is None else n_tayl

    beta = sample.refl_coef_qs()

    j_taylor = _pad_for_broadcasting(
        np.arange(n_tayl), (z_tip, A_tip, n, beta, r_tip, L_tip, g_factor, d_Q0, d_Q1)
    )

    coefs = taylor_coef(
        z_tip, j_taylor, A_tip, n, r_tip, L_tip, g_factor, d_Q0, d_Q1, n_trapz
    )
    delta = np.where(n == 0, 1, 0)
    alpha_eff = np.sum(coefs * beta**j_taylor, axis=0) + delta
    return alpha_eff


def refl_coef_qs(
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
    r"""Return the reflection coefficient corresponding to a particular
    effective polarizability, demodulated at higher harmonics, using a
    Taylor series representation of the bulk FDM.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    A_tip : float
        The tapping amplitude of the AFM tip.
    n : int
        The harmonic of the AFM tip tapping frequency at which to
        demodulate.
    alpha_eff : complex
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
    r_tip, L_tip, g_factor, d_Q0, d_Q1 = _fdm_defaults(
        r_tip, L_tip, g_factor, d_Q0, d_Q1
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

    delta = np.where(n == 0, 1, 0)
    offset_coefs = np.where(j_taylor == 0, delta - alpha_eff_n, coefs)
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
