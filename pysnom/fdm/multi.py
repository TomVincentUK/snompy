import numpy as np
from numpy.polynomial.laguerre import laggauss

from .. import defaults
from .._utils import _fdm_defaults, _pad_for_broadcasting
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


def phi_E_0(z_Q, sample, n_lag=None):
    r"""Return the electric potential and field at the sample surface,
    induced by a charge above a stack of interfaces.

    This function works by performing integrals over all values of in-plane
    electromagnetic wave momentum `q`, using Gauss-Laguerre quadrature.

    Parameters
    ----------
    z_Q : float
        Height of the charge above the sample.
    sample : `~pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate.
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
    :func:`pysnom.sample.Sample.refl_coef_qs`.

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
    # Set defaults
    n_lag = defaults.n_lag if n_lag is None else n_lag

    # Evaluate integral in terms of x = q * 2 * z_Q
    x_lag, w_lag = [
        _pad_for_broadcasting(a, (sample.refl_coef_qs(z_Q),)) for a in laggauss(n_lag)
    ]

    q = x_lag / np.asarray(2 * z_Q)

    beta_q = sample.refl_coef_qs(q)

    phi = np.sum(w_lag * beta_q, axis=0) / (2 * z_Q)
    E = np.sum(w_lag * x_lag * beta_q, axis=0) / (4 * z_Q**2)

    return phi, E


def eff_pos_and_charge(z_Q, sample, n_lag=None):
    r"""Calculate the depth and relative charge of an image charge induced
    below the top surface of a stack of interfaces.

    This function works by evaluating the electric potential and field
    induced at the sample surface using :func:`phi_E_0`.

    Parameters
    ----------
    z_Q : float
        Height of the charge above the sample.
    sample : `~pysnom.sample.Sample`
        Object representing a layered sample with a semi-infinite substrate
        and superstrate.
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
    phi, E = phi_E_0(z_Q, sample, n_lag)
    z_image = np.abs(phi / E) - z_Q
    beta_image = phi**2 / E
    return z_image, beta_image


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
    sample : `~pysnom.sample.Sample`
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
        The order of the Laguerre polynomial used by :func:`phi_E_0`.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    pysnom.fdm.bulk.eff_pol : The bulk equivalent of this function.
    eff_pol_n : The modulated/demodulated version of this function.
    geom_func : Multilayer geometry function.
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
    :func:`geom_func`.

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

    z_q_0 = z_tip + r_tip * d_Q0
    z_im_0, beta_im_0 = eff_pos_and_charge(z_q_0, sample, n_lag)
    f_0 = geom_func(z_tip, z_im_0, r_tip, L_tip, g_factor)

    z_q_1 = z_tip + r_tip * d_Q1
    z_im_1, beta_im_1 = eff_pos_and_charge(z_q_1, sample, n_lag)
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
    sample : `~pysnom.sample.Sample`
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
