import numpy as np

from .._defaults import defaults


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
        beta_0 = beta_1 = sample.image_depth_and_charge(z_Qa, n_lag)

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
        \frac{\ln{\left(
            \frac{4 L_{tip}}{r_{tip} + 4 z_{tip} + 2 r_{tip} d_Q}
        \right)}}
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`d_Q` is `d_Q`, :math:`r_{tip}`
    is `r_tip`, :math:`L_{tip}` is `L_tip`, and :math:`g` is `g_factor`.
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
        \frac{\ln{\left(
            \frac{4 L_{tip}}{r_{tip} + 2 z_{tip} + 2 d_{image}}
        \right)}
        }
        {\ln{\left(\frac{4 L_{tip}}{r_{tip}}\right)}}

    where :math:`z_{tip}` is `z_tip`, :math:`d_{image}` is `d_image`,
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
