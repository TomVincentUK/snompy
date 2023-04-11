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
    z,
    beta,
    radius=defaults["radius"],
    alpha_sphere=4 * np.pi * defaults["radius"] ** 3,
):
    r"""Return the effective probe-sample polarizability using the bulk
    point dipole model.

    Parameters
    ----------
    z : float
        Height of the tip above the sample.
    beta : complex
        Electrostatic reflection coefficient of the interface.
    radius : float
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

        \alpha_{eff} = \frac{\alpha_t}{1 - \left(\frac{\alpha_t \beta}{16 \pi (r + z)^3} \right)}

    where :math:`\alpha_{eff}` is `alpha_eff`, :math:`\alpha_{t}` is
    `alpha_sphere`, :math:`\beta` is `beta`, and :math:`r` is `radius`.
    This is given as equation (14) in reference [1]_.

    References
    ----------
    .. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model
       for quantitative prediction of material contrasts in scattering-type
       near-field optical microscopy,” Opt. Express, vol. 15, no. 14,
       p. 8550, 2007, doi: 10.1364/oe.15.008550.

    """
    return alpha_sphere / (1 - (alpha_sphere * beta / (16 * np.pi * (radius + z) ** 3)))


def eff_pol_n_bulk(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    eps_environment=defaults["eps_environment"],
    beta=None,
    radius=defaults["radius"],
    eps_sphere=None,
    alpha_sphere=None,
):
    r"""Return the effective probe-sample polarizability, demodulated at
    higher harmonics, using the bulk point dipole model.

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
    eps_sphere : complex
        Dielectric function of the sample. Used to calculate
        `alpha_sphere`, and ignored if `alpha_sphere` is specified. If both
        `eps_sphere` and `alpha_sphere` are None, the sphere is assumed to
        be perfectly conducting.
    alpha_sphere : complex
        Polarisability of the conducting sphere used as a model for the AFM
        tip.

    Returns
    -------
    alpha_eff : complex
        Effective polarizability of the tip and sample, demodulated at
        `harmonic`.

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

        \alpha_{t} = 4 \pi r^3 \frac{\varepsilon_t - 1}{\varepsilon_t + 2}

    where :math:`\alpha_{t}` is `alpha_sphere`, :math:`r` is `radius` and
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
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta is None:
            raise ValueError("Either `eps_sample` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_environment, eps_sample)
        else:
            warnings.warn("`beta` overrides `eps_sample` when both are specified.")

    # alpha_sphere calculated from eps_sphere if not specified
    if eps_sphere is None:
        if alpha_sphere is None:
            alpha_sphere = 4 * np.pi * radius**3
    else:
        if alpha_sphere is None:
            alpha_sphere = 4 * np.pi * radius**3 * (eps_sphere - 1) / (eps_sphere + 2)
        else:
            warnings.warn(
                "`alpha_sphere` overrides `eps_sphere` when both are specified."
            )

    # Set oscillation centre  so AFM tip touches sample at z = 0
    z_0 = z + tapping_amplitude

    alpha_eff = demod(
        eff_pol_bulk,
        z_0,
        tapping_amplitude,
        harmonic,
        f_args=(beta, radius, alpha_sphere),
    )

    return alpha_eff
