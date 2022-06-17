"""
References
==========
[1] A. Cvitkovic, N. Ocelic, R. Hillenbrand
    Analytical model for quantitative prediction of material contrasts in
    scattering-type near-field optical microscopy,
    Opt. Express. 15 (2007) 8550.
    https://doi.org/10.1364/oe.15.008550.
"""
import numpy as np
from scipy.integrate import quad


def refl_factor(eps_sample):
    """Electrostatic reflection factor of a sample.
    Defined as beta in equation (2) of reference [1].

    Parameters
    ----------
    eps_sample : complex
        Dielectric function of the sample. Undefined output at -1+0j. Defined
        as epsilon_s in reference [1].

    Returns
    -------
    beta : complex
        Electrostatic reflection factor of the sample.
    """
    eps_sample = eps_sample + 0j  # Cast to complex
    return (eps_sample - 1) / (eps_sample + 1)


def contrast_factor(
    height,
    refl_factor,
    tip_radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
):
    """Dimensionless near-field contrast factor of the tip and sample.
    Defined as eta in equation (12) of reference [1].

    Parameters
    ----------
    height : float
        Height of the tip above the sample. Defined as H in reference [1].
    refl_factor : complex
        Electrostatic reflection factor of the sample. Defined as beta in
        reference [1].
    tip_radius : float
        Radius of curvature of the AFM tip in metres. Defined as R in
        reference [1].
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as L in reference [1].
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge induced
        in the AFM tip to the magnitude of the nearby charge which induced it.
        A small imaginary component can be used to account for phase shifts
        caused by the capacitive interaction of the tip and sample. Defined as
        g in reference [1].

    Returns
    -------
    eta : complex
        Near-field contrast factor of the tip and sample.
    """
    numerator = (
        refl_factor
        * (g_factor - (tip_radius + height) / semi_maj_axis)
        * np.log(4 * semi_maj_axis / (4 * height + 3 * tip_radius))
    )
    denominator = np.log(4 * semi_maj_axis / tip_radius) - refl_factor * (
        g_factor - (4 * height + 3 * tip_radius) / (4 * semi_maj_axis)
    ) * np.log(2 * semi_maj_axis / (2 * height + tip_radius))
    return numerator / denominator


def eff_polarizability(
    height,
    refl_factor,
    tip_radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
):
    """Effective probe-sample polarizability.
    Defined as alpha_eff in equation (13) of reference [1].

    Parameters
    ----------
    height : float
        Height of the tip above the sample. Defined as H in reference [1].
    refl_factor : complex
        Electrostatic reflection factor of the sample. Defined as beta in
        reference [1].
    tip_radius : float
        Radius of curvature of the AFM tip in metres. Defined as R in
        reference [1].
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as L in reference [1].
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge induced
        in the AFM tip to the magnitude of the nearby charge which induced it.
        A small imaginary component can be used to account for phase shifts
        caused by the capacitive interaction of the tip and sample. Defined as
        g in reference [1].

    Returns
    -------
    alpha_eff : complex
        Near-field contrast factor of the tip and sample.
    """
    eta = contrast_factor(height, refl_factor, tip_radius, semi_maj_axis, g_factor)
    multiplier = (
        tip_radius**2
        * semi_maj_axis
        * (
            (2 * semi_maj_axis / tip_radius)
            + np.log(tip_radius / (4 * np.exp(1) * semi_maj_axis))
        )
        / np.log(4 * semi_maj_axis / np.exp(2))
    )
    return multiplier * (2 + eta)


# def contrast_factor_demod(
#     height,
#     refl_factor,
#     harmonic,
#     tap_amp=20e-9,
#     tip_radius=20e-9,
#     semi_maj_axis=300e-9,
#     g_factor=0.7 * np.exp(0.06j),
# ):
#     """Dimensionless near-field contrast factor of the tip and sample,
#     demodulated at the nth harmonic of the tapping amplitude. Defined as eta_n
#     in reference [1].
#
#     Parameters
#     ----------
#     height : float
#         Height of the tip above the sample. Defined as H in reference [1].
#     refl_factor : complex
#         Electrostatic reflection factor of the sample. Defined as beta in
#         reference [1].
#     harmonic: int
#         The harmonic of the AFM tip tapping frequency at which to demodulate.
#     tap_amp: float
#         The tapping amplitude of the AFM tip.
#     tip_radius : float
#         Radius of curvature of the AFM tip in metres. Defined as R in
#         reference [1].
#     semi_maj_axis : float
#         Semi-major axis in metres of the effective spheroid from the FDM.
#         Defined as L in reference [1].
#     g_factor : complex
#         A dimensionless approximation relating the magnitude of charge induced
#         in the AFM tip to the magnitude of the nearby charge which induced it.
#         A small imaginary component can be used to account for phase shifts
#         caused by the capacitive interaction of the tip and sample. Defined as
#         g in reference [1].
#
#     Returns
#     -------
#     eta_n : complex
#         Near-field contrast factor of the tip and sample, demodulated at the
#         nth harmonic of the tapping amplitude.
#     """
#     (
#         height,
#         refl_factor,
#         harmonic,
#         tap_amp,
#         tip_radius,
#         semi_maj_axis,
#         g_factor,
#     ) = np.broadcast_arrays(
#         height, refl_factor, harmonic, tap_amp, tip_radius, semi_maj_axis, g_factor
#     )
#     shape = height.shape
#     eta_n = np.zeros(shape).astype(complex)
#     for inds in np.ndindex(shape):
#         eta_n_real = quad(
#             lambda t: (
#                 contrast_factor(
#                     height + tap_amp * np.sin(t),
#                     refl_factor,
#                     tip_radius,
#                     semi_maj_axis,
#                     g_factor,
#                 )
#                 * np.exp(-1j * harmonic * t)
#             ).real,
#             -np.pi,
#             np.pi,
#         )


# )
# eta_n_imag = quad(
#     lambda t: (
#         contrast_factor(
#             height + tap_amp * np.sin(t),
#             refl_factor,
#             tip_radius,
#             semi_maj_axis,
#             g_factor,
#         )
#         * np.exp(-1j * harmonic * t)
#     ).imag,
#     -np.pi,
#     np.pi,
# )
# return (eta_n_real + eta_n_imag) / (2 * np.pi)


if __name__ == "__main__":
    # import pathlib
    import matplotlib.pyplot as plt

    # from scipy.interpolate import interp1d

    def eps_sho(omega, eps_inf, omega_LO, omega_TO, gamma):
        return eps_inf * (
            1
            + (omega_LO**2 - omega_TO**2)
            / (omega_TO**2 - omega**2 - 1j * gamma * omega)
        )

    def eps_Drude(omega, eps_inf, omega_plasma, gamma):
        return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)

    wavenumber = np.linspace(870e2, 960e2, 512)
    tip_height = 10e-9
    tip_radius = 35e-9
    tap_amp = 25e-9

    # Account for anisotropic SiC dielectric function
    eps_par = eps_sho(wavenumber, 6.78, 782e2, 967e2, 6.6e2)
    eps_perp = eps_sho(wavenumber, 6.56, 797e2, 971e2, 6.6e2)
    beta_par = refl_factor(np.sqrt(eps_par * eps_perp))
    beta_perp = refl_factor(eps_perp)
    beta_SiC = (beta_par + beta_perp) / 2

    eps_Au = eps_Drude(wavenumber, 1, 7.25e4, 2.16e2)
    beta_Au = refl_factor(eps_Au)

    # eta_n_SiC = contrast_factor_demod(tip_height, beta_SiC, 2, tap_amp, tip_radius)
    # eta_n_Au = contrast_factor_demod(tip_height, beta_Au, 2, tap_amp, tip_radius)
    # ratio = eta_n_SiC / eta_n_Au
    # fig, axes = plt.subplots(nrows=2, sharex=True)
    # axes[0].plot(wavenumber * 1e-2, np.abs(ratio))
    # axes[1].plot(wavenumber * 1e-2, np.angle(ratio))
    # fig.tight_layout()
    # plt.show()

    # Dielectric function of Au
    # Au_dir = pathlib.Path("..").joinpath("Au_epsilon")
    # f_SC = Au_dir.joinpath("Olmon_PRB2012_SC.dat")
    # f_EV = Au_dir.joinpath("Olmon_PRB2012_EV.dat")
    # f_TS = Au_dir.joinpath("Olmon_PRB2012_TS.dat")
    #
    # loadtxt_params = dict(skiprows=2, usecols=(1, 2, 3), unpack=True)
    # wavelength_Au, eps1_SC, eps2_SC = np.loadtxt(f_SC, **loadtxt_params)
    # _, eps1_EV, eps2_EV = np.loadtxt(f_EV, **loadtxt_params)
    # _, eps1_TS, eps2_TS = np.loadtxt(f_TS, **loadtxt_params)
    # eps1_Au = np.mean([eps1_SC, eps1_EV, eps1_TS], axis=0)
    # eps2_Au = np.mean([eps2_SC, eps2_EV, eps2_TS], axis=0)
    # wavenumber_Au = 1 / wavelength_Au
    #
    # eps1_func = interp1d(wavenumber_Au, eps1_Au, kind="cubic")
    # eps2_func = interp1d(wavenumber_Au, eps2_Au, kind="cubic")
    # eps_Au = eps1_func(wavenumber) + 1j * eps2_func(wavenumber)

    # alpha_SiC = eff_polarizability(tip_height, beta_SiC, tip_radius)
    # alpha_Au = eff_polarizability(tip_height, beta_Au, tip_radius)
    # alpha_ratio = alpha_SiC / alpha_Au
    #
    # # show_beta = False
    # # if show_beta:
    # #     fig, axes = plt.subplots(nrows=2, sharex=True)
    # #     axes[0].plot(wavenumber, eps_par.real, c="C0", ls="-")
    # #     axes[0].plot(wavenumber, eps_par.imag, c="C0", ls="--")
    # #
    # #     axes[0].plot(wavenumber, eps_perp.real, c="C1", ls="-")
    # #     axes[0].plot(wavenumber, eps_perp.imag, c="C1", ls="--")
    # #
    # #     axes[1].plot(wavenumber, beta_par.real, c="C0", ls="-")
    # #     axes[1].plot(wavenumber, beta_par.imag, c="C0", ls="--")
    # #
    # #     axes[1].plot(wavenumber, beta_perp.real, c="C1", ls="-")
    # #     axes[1].plot(wavenumber, beta_perp.imag, c="C1", ls="--")
    # #
    # #     axes[1].plot(wavenumber, beta_SiC.real, c="C2", ls="-")
    # #     axes[1].plot(wavenumber, beta_SiC.imag, c="C2", ls="--")
    # #
    # #     axes[1].plot(wavenumber, beta_Au.real, c="k", ls="-")
    # #     axes[1].plot(wavenumber, beta_Au.imag, c="k", ls="--")
    # #
    # #     for ax in axes:
    # #         ax.grid(True)
    # #
    # #     fig.tight_layout()
    # #     plt.show()
    # #
    # # show_eps = False
    # # if show_eps:
    # #     fig, axes = plt.subplots(nrows=2, sharex=True)
    # #     axes[0].plot(wavenumber_Au, eps1_Au)
    # #     axes[1].plot(wavenumber_Au, eps2_Au)
    # #
    # #     axes[0].plot(wavenumber, eps_Au_resampled.real)
    # #     axes[1].plot(wavenumber, eps_Au_resampled.imag)
    # #
    # #     axes[0].plot(wavenumber, eps_par.real)
    # #     axes[1].plot(wavenumber, eps_par.imag)
    # #
    # #     axes[0].plot(wavenumber, eps_perp.real)
    # #     axes[1].plot(wavenumber, eps_perp.imag)
    # #
    # #     ax_stretch = 0.1
    # #     axes[0].set(ylim=(300, -6000))
    # #     axes[1].set(
    # #         xlim=(
    # #             wavenumber.min() - ax_stretch * wavenumber.ptp(),
    # #             wavenumber.max() + ax_stretch * wavenumber.ptp(),
    # #         ),
    # #         ylim=(-300, 2500),
    # #     )
    # #
    # #     fig.tight_layout()
    # #     plt.show()
    # #
    # # show_alpha = True
    # # if show_alpha:
