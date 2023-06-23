"""
Point dipole model (:mod:`pysnom.pdm`)
======================================

.. currentmodule:: pysnom.pdm

This module provides functions for simulating the results of scanning
near-field optical microscopy experiments (SNOM) using the point dipole
model (PDM).

Bulk point dipole model
------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    eff_pol_n_bulk
    eff_pol_bulk

"""
import warnings

import numpy as np

from ._defaults import defaults
from .demodulate import demod
from .reflection import refl_coeff


def eff_pol_bulk(
    z_tip,
    beta,
    r_tip=defaults["r_tip"],
    alpha_sphere=4 * np.pi * defaults["r_tip"] ** 3,
):
    r"""Return the effective probe-sample polarizability using the bulk
    point dipole model.

    Parameters
    ----------
    z_tip : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
    r_tip : float
        Radius of curvature of the AFM tip.
    alpha_sphere : complex
        Polarisability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    alpha_eff_0 : complex
        Effective polarizability of the tip and sample.

    See also
    --------
    pysnom.fdm.eff_pol_bulk :
        The finite dipole model (FDM) equivalent of this function.
    eff_pol_n_bulk : The modulated/demodulated version of this function.

    Notes
    -----
    This function implements the equation

    .. math::

        \alpha_{eff} = \frac{\alpha_t}{1 - \left(\frac{\alpha_t \beta}{16 \pi (r_{tip} + z_{tip})^3} \right)}

    where :math:`\alpha_{eff}` is `alpha_eff`, :math:`\alpha_{t}` is
    `alpha_sphere`, :math:`\beta` is `beta`, and :math:`r_{tip}` is `r_tip`.
    This is given as equation (14) in reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.

    """
    return alpha_sphere / (
        1 - (alpha_sphere * beta / (16 * np.pi * (r_tip + z_tip) ** 3))
    )


def eff_pol_n_bulk(
    z_tip,
    A_tip,
    n,
    eps_samp=None,
    eps_env=defaults["eps_env"],
    beta=None,
    r_tip=defaults["r_tip"],
    eps_sphere=None,
    alpha_sphere=None,
    n_trapz=defaults["n_trapz"],
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
    eps_sphere : complex
        Dielectric function of the sample. Used to calculate
        `alpha_sphere`, and ignored if `alpha_sphere` is specified. If both
        `eps_sphere` and `alpha_sphere` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_sphere : complex
        Polarisability of the conducting sphere used as a model for the AFM
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
    pysnom.fdm.eff_pol_n_bulk :
        The finite dipole model equivalent of this function.
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
    # beta calculated from eps_samp if not specified
    if eps_samp is None:
        if beta is None:
            raise ValueError("Either `eps_samp` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_env, eps_samp)
        else:
            warnings.warn("`beta` overrides `eps_samp` when both are specified.")

    # alpha_sphere calculated from eps_sphere if not specified
    if eps_sphere is None:
        if alpha_sphere is None:
            alpha_sphere = 4 * np.pi * r_tip**3
    else:
        if alpha_sphere is None:
            alpha_sphere = 4 * np.pi * r_tip**3 * (eps_sphere - 1) / (eps_sphere + 2)
        else:
            warnings.warn(
                "`alpha_sphere` overrides `eps_sphere` when both are specified."
            )

    # Set oscillation centre  so AFM tip touches sample at z_tip = 0
    z_0 = z_tip + A_tip

    alpha_eff = demod(
        eff_pol_bulk,
        z_0,
        A_tip,
        n,
        f_args=(beta, r_tip, alpha_sphere),
        n_trapz=n_trapz,
    )

    return alpha_eff
