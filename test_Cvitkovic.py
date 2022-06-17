"""
Test of finite dipole model (FDM) by reproducing figure 8 of reference [1].
They didn't report the tip height, so this code plots a range of heights.

References
==========
[1] A. Cvitkovic, N. Ocelic, R. Hillenbrand
    Analytical model for quantitative prediction of material contrasts in
    scattering-type near-field optical microscopy,
    Opt. Express. 15 (2007) 8550.
    https://doi.org/10.1364/oe.15.008550.
[2] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W. Alexander,
    C.A. Ward,
    Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
    Ti, and W in the infrared and far infrared,
    Appl. Opt. 22 (1983) 1099.
    https://doi.org/10.1364/AO.22.001099.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from finite_dipole import refl_coeff, eff_polarizability_nth


def eps_SHO(omega, eps_inf, omega_TO, omega_LO, gamma):
    """
    Single harmonic oscillator dielectric function model. Function definition
    from equation (20) of reference [1].
    """
    return eps_inf * (
        1
        + (omega_LO**2 - omega_TO**2)
        / (omega_TO**2 - omega**2 - 1j * gamma * omega)
    )


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (20) of
    reference [2].
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = np.linspace(800, 1000, 512)[..., np.newaxis] * 1e2
z_0 = np.linspace(0, 100, 21) * 1e-9
tapping_amplitude = 25e-9
radius = 35e-9
harmonic = 3

# Account for anisotropic SiC dielectric function
eps_par = eps_SHO(wavenumber, 6.78, 782e2, 967e2, 6.6e2)  # values from [1]
eps_perp = eps_SHO(wavenumber, 6.56, 797e2, 971e2, 6.6e2)  # values from [1]
beta_par = refl_coeff(np.sqrt(eps_par * eps_perp))
beta_perp = refl_coeff(eps_perp)
beta_SiC = (beta_par + beta_perp) / 2

eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]

alpha_SiC_n, _ = eff_polarizability_nth(
    z_0, tapping_amplitude, harmonic, beta_0=beta_SiC, radius=radius
)
alpha_Au_n, _ = eff_polarizability_nth(
    z_0, tapping_amplitude, harmonic, eps_sample=eps_Au, radius=radius
)

# Plotting
fig = plt.figure()
gs = plt.GridSpec(nrows=2, ncols=2, width_ratios=(1, 0.1))
SM = plt.cm.ScalarMappable(
    cmap=plt.cm.Spectral, norm=Normalize(vmin=z_0.min() * 1e9, vmax=z_0.max() * 1e9)
)

ax_amp = fig.add_subplot(gs[0, 0])
ax_phase = fig.add_subplot(gs[1, 0])
cax = fig.add_subplot(gs[:, -1])

for _z_0, _alpha_SiC_n, _alpha_Au_n in zip(z_0, alpha_SiC_n.T, alpha_Au_n.T):
    c = SM.to_rgba(_z_0 * 1e9)
    ax_amp.plot(wavenumber / 1e2, np.abs(_alpha_SiC_n / _alpha_Au_n), c=c)
    ax_phase.plot(
        wavenumber / 1e2, np.unwrap(np.angle(_alpha_SiC_n / _alpha_Au_n), axis=0), c=c
    )

ax_amp.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    ylabel=r"$\left|\frac{{\alpha}_{"
    f"{harmonic}"
    r", SiC}}{{\alpha}_{"
    f"{harmonic}"
    r", Au}}\right|$",
)
ax_phase.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xlabel=r"${\omega}$ / cm$^{-1}$",
    ylabel=r"$\mathrm{arg}\left(\frac{{\alpha}_{"
    f"{harmonic}"
    r", SiC}}{{\alpha}_{"
    f"{harmonic}"
    r", Au}}\right)$",
)

fig.colorbar(SM, cax=cax, label=r"$z_0$ / nm")

fig.tight_layout()
plt.show()
