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

    eff_pol
    eff_pol_0
    geom_func

Multilayer finite dipole model
------------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_ML
    eff_pol_0_ML
    geom_func_ML
    eff_pos_and_charge
    phi_E_0

"""
import warnings

import numpy as np

from ._defaults import defaults
from .demodulate import demod
from .reflection import interface_stack, refl_coeff, refl_coeff_ML


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

    See also
    --------
    geom_func_ML :
        The multilayer equivalent of this function.

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
    This is given as equation (2) in reference [1]_.

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
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
    x_0=defaults["x_0"],
    x_1=defaults["x_1"],
):
    r"""Return the effective probe-sample polarizability using the bulk
    finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
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
    x_0 : float
        Position of an induced charge 0 within the tip. Specified in units
        of the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified in units
        of the tip radius.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    eff_pol_0_ML :
        The multilayer equivalent of this function.
    eff_pol : The modulated/demodulated version of this function.
    geom_func : Geometry function.

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
    equation (3) in reference [1]_. The function :math:`f_{geom}` is
    implemented here as :func:`geom_func`.

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
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
    x_0=None,
    x_1=defaults["x_1"],
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
    x_0 : float
        Position of an induced charge 0 within the tip. Specified in units
        of the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified in units
        of the tip radius.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `harmonic`.

    See also
    --------
    eff_pol_ML :
        The multilayer equivalent of this function.
    eff_pol_0 : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol_0`.

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
        f_args=(beta, radius, semi_maj_axis, g_factor, x_0, x_1),
    )

    return alpha_eff


def phi_E_0(z_q, beta_stack, t_stack, Laguerre_order=defaults["Laguerre_order"]):
    r"""Return the electric potential and field at the sample surface,
    induced by a charge above a stack of interfaces.

    This function works by performing integrals over all values of in-plane
    electromagnetic wave momentum `k`, using Gauss-Laguerre quadrature.

    Parameters
    ----------
    z_q : float
        Height of the charge above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
    Laguerre_order : int
        The order of the Laguerre polynomial used to evaluate the integrals
        over all `k`.

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
            \phi \rvert_{z=0} &= \int_0^\infty \beta(k) e^{-2 z_q k} dk,
            \ \text{and}\\
            E_z \rvert_{z=0} &= \int_0^\infty \beta(k) k e^{-2 z_q k} dk,
        \end{align*}

    where :math:`\phi` is the electric potential, :math:`E_z` is the
    vertical component of the electric field, :math:`k` is the
    electromagnetic wave momentum, :math:`\beta(k)` is the
    momentum-dependent effective reflection coefficient for the surface,
    and :math:`z_q` is the height of the inducing charge above the
    surface [1]_.

    To do this, it first makes the substitution :math:`x = 2 z_q k`, such
    that the integrals become

    .. math::

        \begin{align*}
            \phi \rvert_{z=0}
            & = \frac{1}{2 z_q} \int_0^\infty
            \beta\left(\frac{x}{2 z_q}\right) e^{-x} dx, \ \text{and}\\
            E_z \rvert_{z=0}
            & = \frac{1}{4 z_q^2} \int_0^\infty
            \beta\left(\frac{x}{2 z_q}\right) x e^{-x} dx.
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
            & \approx \frac{1}{2 z_q}
            \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_q}\right),
            \ \text{and}\\
            E_z \rvert_{z=0}
            & \approx \frac{1}{4 z_q^2}
            \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_q}\right) x_n.
        \end{align*}

    The choice of :math:`N`, defined in this function as `Laguerre_order`,
    will affect the accuracy of the approximation, with higher :math:`N`
    values leading to more accurate evaluation of the integrals.

    In this function the Laguerre weights and roots are found using
    :func:`numpy.polynomial.laguerre.laggauss` and the momentum-dependent
    reflection coefficient is found using
    :func:`pysnom.reflection.refl_coeff_ML`.

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
    # Evaluate integral in terms of x = k * 2 * z_q
    x_Lag, w_Lag = np.polynomial.laguerre.laggauss(Laguerre_order)
    k = x_Lag / np.asarray(2 * z_q)[..., np.newaxis]

    beta_k = refl_coeff_ML(k, beta_stack[..., np.newaxis], t_stack[..., np.newaxis])

    phi = np.sum(w_Lag * beta_k, axis=-1) / (2 * z_q)
    E = np.sum(w_Lag * x_Lag * beta_k, axis=-1) / (4 * z_q**2)

    return phi, E


def eff_pos_and_charge(
    z_q, beta_stack, t_stack, Laguerre_order=defaults["Laguerre_order"]
):
    r"""Calculate the depth and relative charge of an image charge induced
    Below the top surface of a stack of interfaces.

    This function works by evaluating the electric potential and field
    induced at the sample surface using :func:`phi_E_0`.

    Parameters
    ----------
    z_q : float
        Height of the charge above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
    Laguerre_order : int
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
    charge :math:`q` at height :math:`z_q` above a sample surface using the
    equation

    .. math::

        z_{image} = \left|
            \frac{\phi \rvert_{z=0}}{E_z \rvert_{z=0}}
        \right| - z_q,

    and the effective charge of the image, relative to :math:`q`, using the
    equation

    .. math::

        \beta_{image} =
        \frac{ \left( \phi \rvert_{z=0} \right)^2 }
        {E_z \rvert_{z=0}},

    where :math:`\phi` is the electric potential, and :math:`E_z` is the
    vertical component of the electric field. These are based on equations
    (9) and (10) from reference [1]_. The depth :math:`z_q`  is converted
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
    phi, E = phi_E_0(z_q, beta_stack, t_stack, Laguerre_order)
    z_image = np.abs(phi / E) - z_q
    beta_image = phi**2 / E
    return z_image, beta_image


def geom_func_ML(
    z,
    z_image,
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
):
    r"""Return a complex number that encapsulates various geometric
    properties of the tip-sample system for the multilayer finite dipole
    model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    z_image : float
        Depth of an image charge induced below the upper surface of a stack
        of interfaces.
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

    See also
    --------
    geom_func : The bulk equivalent of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        f =
        \left(
            g - \frac{r + z + z_{image}}{2 L}
        \right)
        \frac{\ln{\left(\frac{4 L}{r + 2 z + 2 z_{image}}\right)}}
        {\ln{\left(\frac{4 L}{r}\right)}}

    where :math:`z` is `z`, :math:`z_{image}` is `z_image`, :math:`r` is
    `radius`, :math:`L` is `semi_maj_axis`, and :math:`g` is `g_factor`.
    This is given as equation (11) in reference [1]_.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
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
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
    x_0=defaults["x_0"],
    x_1=defaults["x_1"],
    Laguerre_order=defaults["Laguerre_order"],
):
    r"""Return the effective probe-sample polarizability using the
    multilayer finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    beta_stack : array_like
        Electrostatic reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        Used instead of `eps_stack`, if both are specified.
    t_stack : array_like
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
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
    Laguerre_order : int
        The order of the Laguerre polynomial used by :func:`phi_E_0`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    eff_pol_0 : The bulk equivalent of this function.
    eff_pol_ML : The modulated/demodulated version of this function.
    geom_func_ML : Multilayer geometry function.
    phi_E_0 : Surface electric potential and field.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} =
        1
        + \frac{\beta_{image, 0} f_{geom, ML}(z, z_{image, 0}, r, L, g)}
        {2 (1 - \beta_{image, 1} f_{geom, ML}(z, z_{image, 1}, r, L, g))}

    where :math:`\alpha_{eff}` is `\alpha_eff`; :math:`\beta_{image, i}`
    and :math:`z_{image, i}` are the depth and relative charge of an image
    charge induced by a charge in the tip at :math:`x_{i}`
    (:math:`i=0, 1`), given by `x_0` and `x_1`; :math:`r` is `radius`,
    :math:`L` is `semi_maj_axis`, :math:`g` is `g_factor`, and
    :math:`f_{geom, ML}` is a function encapsulating the geometric
    properties of the tip-sample system for the multilayer finite dipole
    model. This is a modified version of equation (3) from reference [1]_.
    The function :math:`f_{geom, ML}` is implemented here as
    :func:`geom_func_ML`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    z_q_0 = z + radius * x_0
    z_im_0, beta_im_0 = eff_pos_and_charge(z_q_0, beta_stack, t_stack, Laguerre_order)
    f_0 = geom_func_ML(z, z_im_0, radius, semi_maj_axis, g_factor)

    z_q_1 = z + radius * x_1
    z_im_1, beta_im_1 = eff_pos_and_charge(z_q_1, beta_stack, t_stack, Laguerre_order)
    f_1 = geom_func_ML(z, z_im_1, radius, semi_maj_axis, g_factor)

    return 1 + (beta_im_0 * f_0) / (2 * (1 - beta_im_1 * f_1))


def eff_pol_ML(
    z,
    tapping_amplitude,
    harmonic,
    eps_stack=None,
    beta_stack=None,
    t_stack=None,
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
    x_0=None,
    x_1=defaults["x_1"],
    Laguerre_order=defaults["Laguerre_order"],
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the multilayer finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    tapping_amplitude : float
        The tapping amplitude of the AFM tip.
    harmonic : int
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
        Thicknesses of each sandwiched layer between the semi-inifinite
        superstrate and substrate. Must have length one fewer than
        `beta_stack` or two fewer than `eps_stack`. An empty list can be
        used for the case of a single interface.
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
    x_0 : float
        Position of an induced charge 0 within the tip. Specified in units
        of the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified in units
        of the tip radius.
    Laguerre_order : complex
        The order of the Laguerre polynomial used by :func:`phi_E_0`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `harmonic`.

    See also
    --------
    eff_pol : The bulk equivalent of this function.
    eff_pol_0_ML : The unmodulated/demodulated version of this function.
    pysnom.demodulate.demod :
        The function used here for demodulation.

    Notes
    -----
    This function implements
    :math:`\alpha_{eff, n} = \hat{F_n}(\alpha_{eff})`, where
    :math:`\hat{F_n}(\alpha_{eff})` is the :math:`n^{th}` Fourier
    coefficient of the effective polarizability of the tip and sample,
    :math:`\alpha_{eff}`, as described in reference [1]_. The function
    :math:`\alpha_{eff}` is implemented here as :func:`eff_pol_0_ML`.

    References
    ----------
    .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical
       model for scattering infrared near-field microscopy on layered
       systems,” Opt. Express, vol. 20, no. 12, p. 13173, Jun. 2012,
       doi: 10.1364/OE.20.013173.
    """
    if x_0 is None:
        x_0 = 1.31 * semi_maj_axis / (semi_maj_axis + 2 * radius)

    beta_stack, t_stack = interface_stack(eps_stack, beta_stack, t_stack)

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
            radius,
            semi_maj_axis,
            g_factor,
            x_0,
            x_1,
            Laguerre_order,
        ),
    )

    return alpha_eff
