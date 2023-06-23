"""
Finite dipole model (:mod:`pysnom.fdm`)
=======================================

.. currentmodule:: pysnom.fdm

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the finite dipole
model (FDM).

Bulk finite dipole model
------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n_bulk
    eff_pol_bulk
    geom_func_bulk

Taylor series representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated/

    refl_coeff_from_eff_pol_n_bulk_taylor
    eff_pol_n_bulk_taylor
    taylor_coeff_bulk
    geom_func_bulk_taylor

Multilayer finite dipole model
------------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n_multi
    eff_pol_multi
    geom_func_multi
    eff_pos_and_charge
    phi_E_0

"""
import warnings

import numpy as np
from numpy.polynomial import Polynomial, laguerre

from ._defaults import defaults
from .demodulate import demod
from .reflection import interface_stack, refl_coeff, refl_coeff_multi_qs


def geom_func_bulk(
    z_tip,
    d_Q,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
):
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
    geom_func_multi :
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


def eff_pol_bulk(
    z_tip,
    beta,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=defaults["d_Q0"],
    d_Q1=defaults["d_Q1"],
):
    r"""Return the effective probe-sample polarizability using the bulk
    finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
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
    eff_pol_multi :
        The multilayer equivalent of this function.
    eff_pol_n_bulk : The modulated/demodulated version of this function.
    geom_func_bulk : Geometry function.

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
    implemented here as :func:`geom_func_bulk`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    f_0 = geom_func_bulk(z_tip, d_Q0, r_tip, L_tip, g_factor)
    f_1 = geom_func_bulk(z_tip, d_Q1, r_tip, L_tip, g_factor)
    return 1 + (beta * f_0) / (2 * (1 - beta * f_1))


def eff_pol_n_bulk(
    z_tip,
    A_tip,
    n,
    eps_samp=None,
    eps_env=defaults["eps_env"],
    beta=None,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=None,
    d_Q1=defaults["d_Q1"],
    n_trapz=defaults["n_trapz"],
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
    eps_samp : complex
        Dielectric function of the sample. Used to calculate `beta_0`, and
        ignored if `beta_0` is specified.
    eps_env : complex
        Dielectric function of the environment (superstrate). Used to
        calculate `beta_0`, and ignored if `beta_0` is specified.
    beta : complex
        Electrostatic reflection coefficient of the interface.
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
    eff_pol_n_multi :
        The multilayer equivalent of this function.
    eff_pol_bulk : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol_bulk`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    # beta calculated from eps_samp if not specified
    if eps_samp is None:
        if beta is None:
            raise ValueError("Either `eps_samp` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_env, eps_samp)
        else:
            warnings.warn("`beta` overrides `eps_samp` when both are specified.")

    if d_Q0 is None:
        d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)

    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff = demod(
        eff_pol_bulk,
        z_0,
        A_tip,
        n,
        f_args=(beta, r_tip, L_tip, g_factor, d_Q0, d_Q1),
        n_trapz=n_trapz,
    )

    return alpha_eff


def geom_func_bulk_taylor(
    z_tip,
    j_taylor,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=defaults["d_Q0"],
    d_Q1=defaults["d_Q1"],
):
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
    geom_func_bulk
    taylor_coeff_bulk :
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
    :func:`geom_func_bulk`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    f_0 = geom_func_bulk(z_tip, d_Q0, r_tip, L_tip, g_factor)
    f_1 = geom_func_bulk(z_tip, d_Q1, r_tip, L_tip, g_factor)
    return f_0 * f_1 ** (j_taylor - 1)


def taylor_coeff_bulk(
    z_tip,
    j_taylor,
    A_tip,
    n,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=defaults["d_Q0"],
    d_Q1=defaults["d_Q1"],
    n_trapz=defaults["n_trapz"],
):
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
    geom_func_bulk_taylor
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`a_j = \frac{1}{2} \hat{F_n}(f_t)`, where :math:`\hat{F_n}(f_t)`
    is the :math:`n^{th}` Fourier coefficient of the function :math:`f_t`,
    which is implemented here as :func:`geom_func_bulk_taylor`.

    This function returns 0 when :math:`j = 0`, because the Taylor series
    representation of the bulk FDM begins at :math:`j = 1`, however
    :class:`numpy.polynomial.polynomial.Polynomial` requires the first index
    to be zero.
    """
    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    non_zero_terms = (
        demod(
            geom_func_bulk_taylor,
            z_0,
            A_tip,
            n,
            f_args=(j_taylor, r_tip, L_tip, g_factor, d_Q0, d_Q1),
            n_trapz=n_trapz,
        )
        / 2
    )
    return np.where(j_taylor == 0, 0, non_zero_terms)


def eff_pol_n_bulk_taylor(
    z_tip,
    A_tip,
    n,
    eps_samp=None,
    eps_env=defaults["eps_env"],
    beta=None,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=None,
    d_Q1=defaults["d_Q1"],
    n_trapz=defaults["n_trapz"],
    n_tayl=defaults["n_tayl"],
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
    eps_samp : complex
        Dielectric function of the sample. Used to calculate `beta_0`, and
        ignored if `beta_0` is specified.
    eps_env : complex
        Dielectric function of the environment (superstrate). Used to
        calculate `beta_0`, and ignored if `beta_0` is specified.
    beta : complex
        Electrostatic reflection coefficient of the interface.
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
    taylor_coeff_bulk :
        Function that calculates the Taylor series coefficients for the
        bulk FDM.
    eff_pol_n_bulk : The non-Taylor series version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function is valid only for reflection coefficients, `beta`, with
    magnitudes less than around 1. For a more generally applicable function
    use :func:`eff_pol_n_bulk`

    This function implements
    :math:`\alpha_{eff, n} = \delta(n) + \sum_{j=1}^{J} a_j \beta^j`, where
    :math:`\delta` is the Dirac delta function :math:`\beta` is `beta`,
    :math:`j` is the index of the Taylor series, :math:`J` is
    `n_tayl` and :math:`a_j` is the Taylor coefficient, implemented
    here as :func:`taylor_coeff_bulk`.
    """
    # beta calculated from eps_samp if not specified
    if eps_samp is None:
        if beta is None:
            raise ValueError("Either `eps_samp` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_env, eps_samp)
        else:
            warnings.warn("`beta` overrides `eps_samp` when both are specified.")

    if d_Q0 is None:
        d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)

    index_pad_dims = np.max(
        [
            np.ndim(a)
            for a in (
                z_tip,
                A_tip,
                n,
                beta,
                r_tip,
                L_tip,
                g_factor,
                d_Q0,
                d_Q1,
            )
        ]
    )
    j_taylor = np.arange(n_tayl).reshape(-1, *(1,) * index_pad_dims)

    coeffs = taylor_coeff_bulk(
        z_tip,
        j_taylor,
        A_tip,
        n,
        r_tip,
        L_tip,
        g_factor,
        d_Q0,
        d_Q1,
        n_trapz,
    )
    delta = np.where(n == 0, 1, 0)
    alpha_eff = np.sum(coeffs * beta**j_taylor, axis=0) + delta
    return alpha_eff


def refl_coeff_from_eff_pol_n_bulk_taylor(
    z_tip,
    A_tip,
    n,
    alpha_eff_n,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=None,
    d_Q1=defaults["d_Q1"],
    n_trapz=defaults["n_trapz"],
    n_tayl=defaults["n_tayl"],
    beta_threshold=defaults["beta_threshold"],
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
        Electrostatic reflection coefficient of the interface.

    See also
    --------
    taylor_coeff_bulk :
        Function that calculates the Taylor series coefficients for the
        bulk FDM.
    eff_pol_n_bulk_taylor : The inverse of this function.

    Notes
    -----
    This function is valid only `alpha_eff_n` values corresponding to`beta`
    magnitudes less than around 1.

    This function solves, for :math:`\beta`,
    :math:`\alpha_{eff, n} = \delta(n) + \sum_{j=1}^{J} a_j \beta^j`, where
    :math:`\delta` is the Dirac delta function :math:`\beta` is `beta`,
    :math:`j` is the index of the Taylor series, :math:`J` is
    `n_tayl` and :math:`a_j` is the Taylor coefficient, implemented
    here as :func:`taylor_coeff_bulk`.

    There may be multiple possible solutions (or none) for different
    inputs, so this function returns a masked array with first dimension
    whose length is the maximum number of solutions returned for all input
    values. Values which are invalid are masked.
    """
    if d_Q0 is None:
        d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)

    index_pad_dims = np.max(
        [
            np.ndim(a)
            for a in (
                z_tip,
                A_tip,
                n,
                alpha_eff_n,
                r_tip,
                L_tip,
                g_factor,
                d_Q0,
                d_Q1,
            )
        ]
    )
    j_taylor = np.arange(n_tayl).reshape(-1, *(1,) * index_pad_dims)
    coeffs = taylor_coeff_bulk(
        z_tip,
        j_taylor,
        A_tip,
        n,
        r_tip,
        L_tip,
        g_factor,
        d_Q0,
        d_Q1,
        n_trapz,
    )

    delta = np.where(n == 0, 1, 0)
    offset_coeffs = np.where(j_taylor == 0, delta - alpha_eff_n, coeffs)
    all_roots = np.apply_along_axis(lambda c: Polynomial(c).roots(), 0, offset_coeffs)

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


def phi_E_0(z_Q, beta_stack, t_stack, n_lag=defaults["n_lag"]):
    r"""Return the electric potential and field at the sample surface,
    induced by a charge above a stack of interfaces.

    This function works by performing integrals over all values of in-plane
    electromagnetic wave momentum `q`, using Gauss-Laguerre quadrature.

    Parameters
    ----------
    z_Q : float
        Height of the charge above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-infinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
    n_lag : int
        The order of the Laguerre polynomial used to evaluate the integrals
        over all `q`.

    Returns
    -------
    phi : complex
        The electric potential induced at the sample surface by the charge.
    E : complex
        The component of the surface electric field perpendicular to the
        surface.

    See also
    --------
    numpy.polynomial.laguerre.laggauss :
        Laguerre polynomial weights and roots for integration.

    Notes
    -----
    This function evaluates the integrals

    .. math::

        \begin{align*}
            \phi \rvert_{z=0} &= \int_0^\infty \beta(q) e^{-2 z_Q q} dk,
            \ \text{and}\\
            E_z \rvert_{z=0} &= \int_0^\infty \beta(q) q e^{-2 z_Q q} dk,
        \end{align*}

    where :math:`\phi` is the electric potential, :math:`E_z` is the
    vertical component of the electric field, :math:`q` is the
    electromagnetic wave momentum, :math:`\beta(q)` is the
    momentum-dependent effective reflection coefficient for the surface,
    and :math:`z_Q` is the height of the inducing charge above the
    surface [1]_.

    To do this, it first makes the substitution :math:`x = 2 z_Q q`, such
    that the integrals become

    .. math::

        \begin{align*}
            \phi \rvert_{z=0}
            & = \frac{1}{2 z_Q} \int_0^\infty
            \beta\left(\frac{x}{2 z_Q}\right) e^{-x} dx, \ \text{and}\\
            E_z \rvert_{z=0}
            & = \frac{1}{4 z_Q^2} \int_0^\infty
            \beta\left(\frac{x}{2 z_Q}\right) x e^{-x} dx.
        \end{align*}

    It then uses the Gauss-Laguerre approximation [2]_

    .. math::

        \int_0^{\infty} e^{-x} f(x) dx \approx \sum_{n=1}^N w_n f(x_n),

    where :math:`x_n` is the :math:`n^{th}` root of the Laguerre polynomial

    .. math::
        L_N(x) = \sum_{n=0}^{N} {N \choose n} \frac{(-1)^n}{n!} x^n,

    and :math:`w_n` is a weight given by

    .. math::

        w_n = \frac{x_n}{\left((N + 1) L_{N+1}(x_n) \right)^2}.

    The integrals can therefore be approximated by the sums

    .. math::

        \begin{align*}
            \phi \rvert_{z=0}
            & \approx \frac{1}{2 z_Q}
            \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_Q}\right),
            \ \text{and}\\
            E_z \rvert_{z=0}
            & \approx \frac{1}{4 z_Q^2}
            \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_Q}\right) x_n.
        \end{align*}

    The choice of :math:`N`, defined in this function as `n_lag`,
    will affect the accuracy of the approximation, with higher :math:`N`
    values leading to more accurate evaluation of the integrals.

    In this function the Laguerre weights and roots are found using
    :func:`numpy.polynomial.laguerre.laggauss` and the momentum-dependent
    reflection coefficient is found using
    :func:`pysnom.reflection.refl_coeff_multi_qs`.

    References
    ----------
    .. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
       R. Hillenbrand, “Subsurface chemical nanoidentification by nano-FTIR
       spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359, Dec. 2020,
       doi: 10.1038/s41467-020-17034-6.
    .. [2] S. Ehrich, “On stratified extensions of Gauss-Laguerre and
       Gauss-Hermite quadrature formulas,” J. Comput. Appl. Math., vol.
       140, no. 1-2, pp. 291-299, Mar. 2002,
       doi: 10.1016/S0377-0427(01)00407-1.
    """
    # Evaluate integral in terms of x = q * 2 * z_Q
    x_lag, w_lag = laguerre.laggauss(n_lag)
    q = x_lag / np.asarray(2 * z_Q)[..., np.newaxis]

    beta_q = refl_coeff_multi_qs(
        q, beta_stack[..., np.newaxis], t_stack[..., np.newaxis]
    )

    phi = np.sum(w_lag * beta_q, axis=-1) / (2 * z_Q)
    E = np.sum(w_lag * x_lag * beta_q, axis=-1) / (4 * z_Q**2)

    return phi, E


def eff_pos_and_charge(z_Q, beta_stack, t_stack, n_lag=defaults["n_lag"]):
    r"""Calculate the depth and relative charge of an image charge induced
    below the top surface of a stack of interfaces.

    This function works by evaluating the electric potential and field
    induced at the sample surface using :func:`phi_E_0`.

    Parameters
    ----------
    z_Q : float
        Height of the charge above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-infinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
    n_lag : int
        The order of the Laguerre polynomial used by :func:`phi_E_0`.

    Returns
    -------
    phi : complex
        The electric potential induced at the sample surface by a the
        charge.
    E : complex
        The component of the surface electric field perpendicular to the
        surface.

    See also
    --------
    phi_E_0 : Surface electric potential and field.

    Notes
    -----

    This function calculates the depth of an image charge induced by a
    charge :math:`q` at height :math:`z_Q` above a sample surface using the
    equation

    .. math::

        z_{image} = \left|
            \frac{\phi \rvert_{z=0}}{E_z \rvert_{z=0}}
        \right| - z_Q,

    and the effective charge of the image, relative to :math:`q`, using the
    equation

    .. math::

        \beta_{image} =
        \frac{ \left( \phi \rvert_{z=0} \right)^2 }
        {E_z \rvert_{z=0}},

    where :math:`\phi` is the electric potential, and :math:`E_z` is the
    vertical component of the electric field. These are based on equations
    (9) and (10) from reference [1]_. The depth :math:`z_Q`  is converted
    to a real number by taking the absolute value of the
    :math:`\phi`-:math:`E_z` ratio, as described in reference [2]_.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    .. [2] C. Lupo et al., “Quantitative infrared near-field imaging of
       suspended topological insulator nanostructures,” pp. 1–23, Dec.
       2021, [Online]. Available: http://arxiv.org/abs/2112.10104
    """
    phi, E = phi_E_0(z_Q, beta_stack, t_stack, n_lag)
    z_image = np.abs(phi / E) - z_Q
    beta_image = phi**2 / E
    return z_image, beta_image


def geom_func_multi(
    z_tip,
    d_image,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
):
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
    geom_func_bulk : The bulk equivalent of this function.

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


def eff_pol_multi(
    z_tip,
    beta_stack=None,
    t_stack=None,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=defaults["d_Q0"],
    d_Q1=defaults["d_Q1"],
    n_lag=defaults["n_lag"],
):
    r"""Return the effective probe-sample polarizability using the
    multilayer finite dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-infinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
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
        The order of the Laguerre polynomial used by :func:`phi_E_0`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    eff_pol_bulk : The bulk equivalent of this function.
    eff_pol_n_multi : The modulated/demodulated version of this function.
    geom_func_multi : Multilayer geometry function.
    phi_E_0 : Surface electric potential and field.

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
    :func:`geom_func_multi`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    z_q_0 = z_tip + r_tip * d_Q0
    z_im_0, beta_im_0 = eff_pos_and_charge(z_q_0, beta_stack, t_stack, n_lag)
    f_0 = geom_func_multi(z_tip, z_im_0, r_tip, L_tip, g_factor)

    z_q_1 = z_tip + r_tip * d_Q1
    z_im_1, beta_im_1 = eff_pos_and_charge(z_q_1, beta_stack, t_stack, n_lag)
    f_1 = geom_func_multi(z_tip, z_im_1, r_tip, L_tip, g_factor)

    return 1 + (beta_im_0 * f_0) / (2 * (1 - beta_im_1 * f_1))


def eff_pol_n_multi(
    z_tip,
    A_tip,
    n,
    eps_stack=None,
    beta_stack=None,
    t_stack=None,
    r_tip=defaults["r_tip"],
    L_tip=defaults["L_tip"],
    g_factor=defaults["g_factor"],
    d_Q0=None,
    d_Q1=defaults["d_Q1"],
    n_lag=defaults["n_lag"],
    n_trapz=defaults["n_trapz"],
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
    eps_stack : array_like
        Dielectric functions of each layer in the stack. Layers should be
        arranged from the top down, starting with the semi-infinite
        superstrate and ending with the semi-infinite substrate. Ignored
        if `beta_stack` is specified.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-infinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
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
        The order of the Laguerre polynomial used by :func:`phi_E_0`.
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
    eff_pol_n_bulk : The bulk equivalent of this function.
    eff_pol_multi : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol_multi`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    if d_Q0 is None:
        d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)

    beta_stack, t_stack = interface_stack(eps_stack, beta_stack, t_stack)

    # Set oscillation centre so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff = demod(
        eff_pol_multi,
        z_0,
        A_tip,
        n,
        f_args=(
            beta_stack,
            t_stack,
            r_tip,
            L_tip,
            g_factor,
            d_Q0,
            d_Q1,
            n_lag,
        ),
        n_trapz=n_trapz,
    )

    return alpha_eff
