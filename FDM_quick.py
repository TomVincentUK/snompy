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
[3] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W. Alexander,
    C.A. Ward,
    Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
    Ti, and W in the infrared and far infrared,
    Appl. Opt. 22 (1983) 1099.
    https://doi.org/10.1364/AO.22.001099.
"""
import warnings
import numpy as np
from numba import njit
from scipy.integrate import quad


def complex_quad(func, a, b, **kwargs):
    """
    Wrapper to `scipy.integrate.quad` to allow complex integrands.
    """
    real_part = quad(lambda t, *args: np.real(func(t, *args)), a, b, **kwargs)
    imag_part = quad(lambda t, *args: np.imag(func(t, *args)), a, b, **kwargs)
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


@njit
def eff_polarizability(z, beta_0, beta_1, X_0, X_1, radius, semi_maj_axis, g_factor):
    """
    This is the Hauer version.
    """
    f_0 = geometry_function(z, X_0, radius, semi_maj_axis, g_factor)
    f_1 = geometry_function(z, X_1, radius, semi_maj_axis, g_factor)
    return 1 + (beta_0 * f_0) / (2 * (1 - beta_1 * f_1))


@njit
def Fourier_envelope(t, n):
    return np.exp(-1j * n * t)


@njit
def _integrand(
    t,
    z,
    tapping_amplitude,
    harmonic,
    beta_0,
    beta_1,
    X_0,
    X_1,
    radius,
    semi_maj_axis,
    g_factor,
):
    alpha_eff = eff_polarizability(
        z + tapping_amplitude * (1 + np.cos(t)),
        beta_0,
        beta_1,
        X_0,
        X_1,
        radius,
        semi_maj_axis,
        g_factor,
    )
    sinusoids = Fourier_envelope(t, harmonic)
    return alpha_eff * sinusoids


def _integral(
    z,
    tapping_amplitude,
    harmonic,
    beta_0,
    beta_1,
    X_0,
    X_1,
    radius,
    semi_maj_axis,
    g_factor,
):
    return complex_quad(
        _integrand,
        -np.pi,
        np.pi,
        args=(
            z,
            tapping_amplitude,
            harmonic,
            beta_0,
            beta_1,
            X_0,
            X_1,
            radius,
            semi_maj_axis,
            g_factor,
        ),
    )


_integral_vec = np.vectorize(_integral)


def eff_polarizability_nth(
    z,
    tapping_amplitude,
    harmonic,
    eps_sample=None,
    beta_0=None,
    beta_1=None,
    X_0=1.31,
    X_1=0.5,
    radius=20e-9,
    semi_maj_axis=300e-9,
    g_factor=0.7 * np.exp(0.06j),
):
    # beta calculated from eps_sample if not specified
    if eps_sample is None:
        if beta_0 is None:
            raise ValueError("Either `eps_sample` or `beta_0` must be specified.")
    else:
        if beta_0 is None:
            beta_0 = refl_coeff(eps_sample)
        else:
            warnings.warn("`beta_0` overrides `eps_sample` when both are specified.")

    # Assume only one beta value unless both specified (for multilayer FDM)
    if beta_1 is None:
        beta_1 = beta_0

    alpha_eff, alpha_eff_err = _integral_vec(
        z,
        tapping_amplitude,
        harmonic,
        beta_0,
        beta_1,
        X_0,
        X_1,
        radius,
        semi_maj_axis,
        g_factor,
    )
    return alpha_eff / (2 * np.pi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def eps_sho(omega, eps_inf, omega_TO, omega_LO, gamma):
        """Single harmonic oscillator"""
        return eps_inf * (
            1
            + (omega_LO**2 - omega_TO**2)
            / (omega_TO**2 - omega**2 - 1j * gamma * omega)
        )

    def eps_Drude(omega, eps_inf, omega_plasma, gamma):
        return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)

    wavenumber = np.linspace(800, 1000, 512)[..., np.newaxis] * 1e2

    # Account for anisotropic SiC dielectric function
    eps_par = eps_sho(wavenumber, 6.78, 782e2, 967e2, 6.6e2)  # values from [2]
    eps_perp = eps_sho(wavenumber, 6.56, 797e2, 971e2, 6.6e2)  # values from [2]
    beta_par = refl_coeff(np.sqrt(eps_par * eps_perp))
    beta_perp = refl_coeff(eps_perp)
    beta_SiC = (beta_par + beta_perp) / 2

    eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [3]

    z_0 = np.linspace(0, 100, 11) * 1e-9
    tapping_amplitude = 25e-9
    radius = 35e-9
    harmonic = 3
    alpha_SiC_n = eff_polarizability_nth(
        z_0, tapping_amplitude, harmonic, beta_0=beta_SiC, radius=radius
    )
    alpha_Au_n = eff_polarizability_nth(
        z_0, tapping_amplitude, harmonic, eps_sample=eps_Au, radius=radius
    )

    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(wavenumber / 1e2, np.abs(alpha_SiC_n / alpha_Au_n))
    axes[1].plot(
        wavenumber / 1e2,
        np.unwrap(np.angle(alpha_SiC_n / alpha_Au_n), axis=0),
    )

    axes[0].set(
        ylabel=r"$\left|\frac{{\alpha}_{"
        f"{harmonic}"
        r", SiC}}{{\alpha}_{"
        f"{harmonic}"
        r", Au}}\right|$"
    )
    axes[1].set(
        xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
        xlabel=r"${\omega}$ / cm$^{-1}$",
        ylabel=r"$\mathrm{arg}\left(\frac{{\alpha}_{"
        f"{harmonic}"
        r", SiC}}{{\alpha}_{"
        f"{harmonic}"
        r", Au}}\right)$",
    )
    fig.tight_layout()
    plt.show()
