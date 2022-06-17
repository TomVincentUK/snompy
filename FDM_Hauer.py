"""
References
==========
[1] B. Hauer, A.P. Engelhardt, T. Taubner,
    Quasi-analytical model for scattering infrared near-field microscopy on
    layered systems,
    Opt. Express. 20 (2012) 13173.
    https://doi.org/10.1364/OE.20.013173.
[2] A. Cvitkovic, N. Ocelic, R. Hillenbrand
    Analytical model for quantitative prediction of material contrasts in
    scattering-type near-field optical microscopy,
    Opt. Express. 15 (2007) 8550.
    https://doi.org/10.1364/oe.15.008550.
"""
import warnings
import numpy as np
from numba import njit
from scipy.integrate import quad
from tqdm import tqdm


def tqdm_nd(shape, **kwargs):
    """
    Creates an n-dimensional tqdm iterator. Iterator yields tuple of indices.
    kwargs passed to tqdm.
    """
    return tqdm(list(np.ndindex(shape)), **kwargs)


def complex_quad(func, a, b, **kwargs):
    real_part = quad(lambda t: np.real(func(t)), a, b, **kwargs)
    imag_part = quad(lambda t: np.imag(func(t)), a, b, **kwargs)
    return real_part[0] + 1j * imag_part[0], real_part[1] + 1j * imag_part[1]


@njit
def refl_coeff(eps_i, eps_j=1 + 0j):
    return (eps_i - eps_j) / (eps_i + eps_j) + 0j  # + 0j ensures complex


@njit
def geometry_function(z, charge_pos, radius, semi_maj_axis, g_factor):
    return (
        (g_factor - (radius + 2 * z + charge_pos * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 4 * z + 2 * charge_pos * radius))
        / np.log(4 * semi_maj_axis / radius)
    )


# def eff_polarizability(
#     z,
#     eps_sample=None,
#     beta=None,
#     radius=20e-9,
#     semi_maj_axis=300e-9,
#     g_factor=0.7 * np.exp(0.06j),
#     charge_positions=(1.31, 0.5),
# ):
#     """
#     This is the Hauer version.
#     """
#     # beta calculated from eps_sample if not specified
#     if eps_sample is None:
#         if beta is None:
#             raise ValueError("Either `eps_sample` or `beta` must be specified.")
#     else:
#         if beta is None:
#             beta = refl_coeff(eps_sample)
#         else:
#             warnings.warn("`beta` overrides `eps_sample` when both are specified.")
#
#     f_0 = geometry_function(z, charge_positions[0], radius, semi_maj_axis, g_factor)
#     f_1 = geometry_function(z, charge_positions[1], radius, semi_maj_axis, g_factor)
#     return 1 + (beta * f_0) / (2 * (1 - beta * f_1))


def eff_polarizability(
    z,
    eps_sample=None,
    beta=None,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    charge_positions=(1.31, 0.5),
):
    """
    This is the Cvitovic version.
    """
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta is None:
            raise ValueError("Either `eps_sample` or `beta` must be specified.")
    else:
        if beta is None:
            beta = refl_coeff(eps_sample)
        else:
            warnings.warn("`beta` overrides `eps_sample` when both are specified.")

    outside = (
        radius**2
        * semi_maj_axis
        * (
            2 * semi_maj_axis / radius
            + np.log(radius / (4 * np.exp(1) * semi_maj_axis))
        )
        / np.log(4 * semi_maj_axis / np.exp(2))
    )
    numerator = (
        beta
        * (g_factor - (radius + z / semi_maj_axis))
        * np.log(4 * semi_maj_axis / (4 * z + 3 * radius))
    )
    denominator = np.log(4 * semi_maj_axis / radius) - beta * (
        g_factor - (3 * radius + 4 * z) / (4 * semi_maj_axis)
    ) * np.log(2 * semi_maj_axis / (2 * z + radius))
    return outside * (2 + numerator / denominator)


def eff_polarizability_nth(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    beta=None,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    charge_positions=(1.31, 0.5),
):
    (
        z,
        tapping_amplitude,
        harmonic,
        eps_sample,
        beta,
        radius,
        semi_maj_axis,
        g_factor,
    ) = np.broadcast_arrays(
        z,
        tapping_amplitude,
        harmonic,
        eps_sample,
        beta,
        radius,
        semi_maj_axis,
        g_factor,
    )
    shape = z.shape
    ndim = z.ndim
    charge_positions = np.ones((*shape, 2)) * np.reshape(
        charge_positions, (*ndim * (1,), -1)
    )
    alpha_eff = np.zeros(shape) + 0j
    for inds in tqdm_nd(shape):

        def _integrand(t):
            alpha_eff = eff_polarizability(
                z[inds] + tapping_amplitude[inds] * (1 + np.cos(t)),
                eps_sample[inds],
                beta[inds],
                radius[inds],
                semi_maj_axis[inds],
                g_factor[inds],
                charge_positions[inds],
            )
            sinusoids = np.exp(-1j * harmonic[inds] * t)
            return alpha_eff * sinusoids

        alpha_eff[inds], _ = complex_quad(_integrand, -np.pi, np.pi)

    return alpha_eff / (2 * np.pi)


if __name__ == "__main__":
    import pathlib
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    def eps_sho(omega, eps_inf, omega_TO, omega_LO, gamma):
        return eps_inf * (
            1
            + (omega_LO**2 - omega_TO**2)
            / (omega_TO**2 - omega**2 - 1j * gamma * omega)
        )

    def eps_Drude(omega, eps_inf, omega_plasma, gamma):
        return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)

    wavenumber = np.linspace(850, 1000, 512) * 1e2

    # Account for anisotropic SiC dielectric function
    eps_par = eps_sho(wavenumber, 6.78, 782e2, 967e2, 6.6e2)
    eps_perp = eps_sho(wavenumber, 6.56, 797e2, 971e2, 6.6e2)
    beta_par = refl_coeff(np.sqrt(eps_par * eps_perp))
    beta_perp = refl_coeff(eps_perp)
    beta_SiC = (beta_par + beta_perp) / 2

    empirical_Au = False
    if empirical_Au:
        # Dielectric function of Au
        Au_dir = pathlib.Path("..").joinpath("Au_epsilon")
        f_SC = Au_dir.joinpath("Olmon_PRB2012_SC.dat")
        f_EV = Au_dir.joinpath("Olmon_PRB2012_EV.dat")
        f_TS = Au_dir.joinpath("Olmon_PRB2012_TS.dat")

        loadtxt_params = dict(skiprows=2, usecols=(1, 2, 3), unpack=True)
        wavelength_Au, eps1_SC, eps2_SC = np.loadtxt(f_SC, **loadtxt_params)
        _, eps1_EV, eps2_EV = np.loadtxt(f_EV, **loadtxt_params)
        _, eps1_TS, eps2_TS = np.loadtxt(f_TS, **loadtxt_params)
        eps1_Au = np.mean([eps1_SC, eps1_EV, eps1_TS], axis=0)
        eps2_Au = np.mean([eps2_SC, eps2_EV, eps2_TS], axis=0)
        wavenumber_Au = 1 / wavelength_Au

        eps1_func = interp1d(wavenumber_Au, eps1_Au, kind="cubic")
        eps2_func = interp1d(wavenumber_Au, eps2_Au, kind="cubic")
        eps_Au = eps1_func(wavenumber) + 1j * eps2_func(wavenumber)
    else:
        eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)

    z_0 = 1e-9
    tapping_amplitude = 20e-9
    harmonic = 2

    fig, axes = plt.subplots(nrows=2, sharex=True)
    alpha_SiC_n = eff_polarizability_nth(
        z_0, tapping_amplitude, harmonic, beta=beta_SiC, radius=35e-9
    )
    alpha_Au_n = eff_polarizability_nth(
        z_0, tapping_amplitude, harmonic, eps_sample=eps_Au, radius=35e-9
    )
    axes[0].plot(wavenumber / 1e2, np.abs(alpha_SiC_n / alpha_Au_n))
    axes[1].plot(
        wavenumber / 1e2,
        np.angle(alpha_SiC_n / alpha_Au_n) % (2 * np.pi),
    )
    fig.tight_layout()
    plt.show()
