"""
Multilayer finite dipole model (:mod:`finite_dipole.multilayer`)
================================================================

.. currentmodule:: finite_dipole.multilayer

WRITE A DESCRIPTION HERE.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    phi_E_0
    eff_pos_and_charge
    geom_func_ML
    eff_pol_0_ML
    eff_pol_ML
"""
import numpy as np

from ._defaults import defaults
from .demodulate import demod
from .reflection import interface_stack, refl_coeff_ML


def phi_E_0(z_q, beta_stack, t_stack, Laguerre_order=defaults["Laguerre_order"]):
    """Write me!

    Calculate phi and E using Gauss-Laguerre quadrature"""
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
    """Write me."""
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
    properties of the tip-sample system for the multilayer FDM.

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
        Semi-major axis length of the effective spheroid from the FDM.
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

        f =
        \left(
            g - \frac{r + z + z_{image}}{2 L}
        \right)
        \frac{\ln{\left(\frac{4 L}{r + 2 z + 2 z_{image}}\right)}}
        {\ln{\left(\frac{4 L}{r}\right)}}

    where :math:`z` is `z`, :math:`z_{image}` is `z_image`, :math:`r` is
    `radius`, :math:`L` is `semi_maj_axis`, and :math:`g` is `g_factor`.
    This is given as equation (11) in reference_[1].

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
    x_0=defaults["x_0"],
    x_1=defaults["x_1"],
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
    Laguerre_order=defaults["Laguerre_order"],
):
    r"""Return the effective probe-sample polarizability using the
    multilayer finite dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
    x_0 : float
        Position of an induced charge 0 within the tip. Specified relative
        to the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified relative
        to the tip radius.
    radius : float
        Radius of curvature of the AFM tip.
    semi_maj_axis : float
        Semi-major axis length of the effective spheroid from the FDM.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.
    Laguerre_order : int
        Write me!

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} =
        1
        + \frac{\beta_{image, 0} f_{geom, ML}(z, z_{image, 0}, r, L, g)}
        {2 (1 - \beta_{image, 1} f_{geom, ML}(z, z_{image, 1}, r, L, g))}

    where :math:`\alpha_{eff}` is `\alpha_eff`; :math:`\beta_{image, i}`
    and z_{image, i} are the depth and relative charge of an image charge
    induced by a charge in the tip at :math:`x_{i}` (:math:`i=0, 1`), given
    by `x_0` and `x_1`; :math:`r` is `radius`, :math:`L` is
    `semi_maj_axis`, :math:`g` is `g_factor`, and :math:`f_{geom, ML}` is a
    function encapsulating the geometric properties of the tip-sample
    system for the multilayer finite dipole model. This is a modified
    version of equation (3) from reference_[1]. The function
    :math:`f_{geom, ML}` is implemented here as
    `finite_dipole.bulk.geom_func_ML`.

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
    x_0=None,
    x_1=defaults["x_1"],
    radius=defaults["radius"],
    semi_maj_axis=defaults["semi_maj_axis"],
    g_factor=defaults["g_factor"],
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
    x_0 : float
        Position of an induced charge 0 within the tip. Specified relative
        to the tip radius.
    x_1 : float
        Position of an induced charge 1 within the tip. Specified relative
        to the tip radius.
    radius : float
        Radius of curvature of the AFM tip.
    semi_maj_axis : float
        Semi-major axis length of the effective spheroid from the FDM.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.
    Laguerre_order : int
        Write me!

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
    `finite_dipole.bulk.eff_pol_0_ML`.

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
            x_0,
            x_1,
            radius,
            semi_maj_axis,
            g_factor,
            Laguerre_order,
        ),
    )

    return alpha_eff
