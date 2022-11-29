"""
A quick comparison script that I'll delete in a bit.

References
----------
.. [1] A. Cvitkovic, N. Ocelic, R. Hillenbrand
   Analytical model for quantitative prediction of material contrasts in
   scattering-type near-field optical microscopy,
   Opt. Express. 15 (2007) 8550.
   https://doi.org/10.1364/oe.15.008550.
.. [2] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W.
   Alexander, C.A. Ward,
   Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
   Ti, and W in the infrared and far infrared,
   Appl. Opt. 22 (1983) 1099.
   https://doi.org/10.1364/AO.22.001099.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from finite_dipole import eff_pol
from finite_dipole.tools import refl_coeff


def eps_SHO(omega, eps_inf, omega_TO, omega_LO, gamma):
    """
    Single harmonic oscillator dielectric function model. Function
    definition from equation (20) of reference [1]_.
    """
    return eps_inf * (
        1
        + (omega_LO**2 - omega_TO**2)
        / (omega_TO**2 - omega**2 - 1j * gamma * omega)
    )


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (2)
    of reference [2]_.
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = np.linspace(885, 985, 512)[..., np.newaxis] * 1e2
z_0 = 10e-9
tapping_amplitude = 25e-9
radius = 35e-9
harmonic = 2

# Account for anisotropic SiC dielectric function
eps_par = eps_SHO(wavenumber, 6.78, 782e2, 967e2, 6.6e2)  # values from [1]_
eps_perp = eps_SHO(wavenumber, 6.56, 797e2, 971e2, 6.6e2)  # values from [1]_
beta_par = refl_coeff(1 + 0j, np.sqrt(eps_par * eps_perp))
beta_perp = refl_coeff(1 + 0j, eps_perp)
beta_SiC = (beta_par + beta_perp) / 2

eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]_
beta_Au = refl_coeff(1 + 0j, eps_Au)  # Not used except for comparison

alpha_SiC_n = eff_pol(z_0, tapping_amplitude, harmonic, beta=beta_SiC, radius=radius)
alpha_Au_n = eff_pol(z_0, tapping_amplitude, harmonic, eps_sample=eps_Au, radius=radius)

# Plotting defaults
c_real = "C0"
c_imag = "C1"
c_amp = "C2"
c_phase = "C3"
ls_par = "--"
ls_perp = ":"
ls_full = "-"

# Plotting for SiC
fig_SiC, (ax_eps_SiC, ax_beta_SiC, ax_alpha_SiC) = plt.subplots(nrows=3, sharex=True)

ax_eps_SiC.plot(wavenumber * 1e-2, eps_par.real, c=c_real, ls=ls_par, label="real")
ax_eps_SiC.plot(wavenumber * 1e-2, eps_par.imag, c=c_imag, ls=ls_par, label="imaginary")
ax_eps_SiC.plot(wavenumber * 1e-2, eps_perp.real, c=c_real, ls=ls_perp)
ax_eps_SiC.plot(wavenumber * 1e-2, eps_perp.imag, c=c_imag, ls=ls_perp)
ax_eps_SiC.legend()
ax_eps_SiC.set_ylabel(r"${\varepsilon}_{SiC}$")

ax_beta_SiC.plot(
    wavenumber * 1e-2, beta_par.real, c=c_real, ls=ls_par, label="parallel"
)
ax_beta_SiC.plot(wavenumber * 1e-2, beta_par.imag, c=c_imag, ls=ls_par)
ax_beta_SiC.plot(
    wavenumber * 1e-2, beta_perp.real, c=c_real, ls=ls_perp, label="perpendicular"
)
ax_beta_SiC.plot(wavenumber * 1e-2, beta_perp.imag, c=c_imag, ls=ls_perp)
ax_beta_SiC.plot(wavenumber * 1e-2, beta_SiC.real, c=c_real, ls=ls_full, label="full")
ax_beta_SiC.plot(wavenumber * 1e-2, beta_SiC.imag, c=c_imag, ls=ls_full)
ax_beta_SiC.legend()
ax_beta_SiC.set_ylabel(r"${\beta}_{SiC}$")

(amp_SiC,) = ax_alpha_SiC.plot(
    wavenumber * 1e-2, np.abs(alpha_SiC_n), c=c_amp, ls=ls_full, label="amplitude"
)
ax_alpha_SiC.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xlabel=r"$\omega$ / cm$^{-1}$",
    ylabel=r"$\left|{\alpha}_{eff, " f"{harmonic}" r", SiC}\right|$",
)

ax_phase_SiC = ax_alpha_SiC.twinx()
(phase_SiC,) = ax_phase_SiC.plot(
    wavenumber * 1e-2,
    np.unwrap(np.angle(alpha_SiC_n), axis=0),
    c=c_phase,
    ls=ls_full,
    label="phase",
)
ax_phase_SiC.set_ylabel(
    r"$\mathrm{arg}\left({\alpha}_{eff, " f"{harmonic}" r", SiC}\right)$"
)
ax_phase_SiC.legend(handles=(amp_SiC, phase_SiC))

fig_SiC.tight_layout()

# Plotting for Au
fig_Au, (ax_eps_Au, ax_beta_Au, ax_alpha_Au) = plt.subplots(nrows=3, sharex=True)

ax_eps_Au.plot(wavenumber * 1e-2, eps_Au.real, c=c_real, ls=ls_full, label="real")
ax_eps_Au.plot(wavenumber * 1e-2, eps_Au.imag, c=c_imag, ls=ls_full, label="imaginary")
ax_eps_Au.legend()
ax_eps_Au.set_ylabel(r"${\varepsilon}_{Au}$")

ax_beta_Au.plot(wavenumber * 1e-2, beta_Au.real, c=c_real, ls=ls_full)
ax_beta_Au.plot(wavenumber * 1e-2, beta_Au.imag, c=c_imag, ls=ls_full)
ax_beta_Au.set_ylabel(r"${\beta}_{Au}$")

(amp_Au,) = ax_alpha_Au.plot(
    wavenumber * 1e-2, np.abs(alpha_Au_n), c=c_amp, ls=ls_full, label="amplitude"
)
ax_alpha_Au.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xlabel=r"$\omega$ / cm$^{-1}$",
    ylabel=r"$\left|{\alpha}_{eff, " f"{harmonic}" r", Au}\right|$",
)

ax_phase_Au = ax_alpha_Au.twinx()
(phase_Au,) = ax_phase_Au.plot(
    wavenumber * 1e-2,
    np.unwrap(np.angle(alpha_Au_n), axis=0),
    c=c_phase,
    ls=ls_full,
    label="phase",
)
ax_phase_Au.set_ylabel(
    r"$\mathrm{arg}\left({\alpha}_{eff, " f"{harmonic}" r", Au}\right)$"
)
ax_phase_Au.legend(handles=(amp_Au, phase_Au))

fig_Au.tight_layout()


# Plotting the ratio
fig_norm, ax_norm = plt.subplots()

(amp_norm,) = ax_norm.plot(
    wavenumber * 1e-2,
    np.abs(alpha_SiC_n / alpha_Au_n),
    c=c_amp,
    ls=ls_full,
    label="amplitude",
)
ax_norm.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xlabel=r"$\omega$ / cm$^{-1}$",
    ylabel=r"$\left|\frac{{\alpha}_{eff, "
    f"{harmonic}"
    r", SiC}}{{\alpha}_{eff, "
    f"{harmonic}"
    r", Au}}\right|$",
)

ax_phase_norm = ax_norm.twinx()
(phase_norm,) = ax_phase_norm.plot(
    wavenumber * 1e-2,
    np.unwrap(np.angle(alpha_SiC_n / alpha_Au_n), axis=0),
    c=c_phase,
    ls=ls_full,
    label="phase",
)
ax_phase_norm.set_ylabel(
    r"$\mathrm{arg}\left(\frac{{\alpha}_{eff, "
    f"{harmonic}"
    r", SiC}}{{\alpha}_{eff, "
    f"{harmonic}"
    r", Au}}\right)$"
)
ax_phase_norm.legend(handles=(amp_norm, phase_norm))

fig_norm.tight_layout()

plt.show(block=False)

# gs = plt.GridSpec(nrows=2, ncols=2, width_ratios=(1, 0.1))
# SM = plt.cm.ScalarMappable(
#     cmap=plt.cm.Spectral, norm=Normalize(vmin=z_0.min() * 1e9, vmax=z_0.max() * 1e9)
# )
#
# ax_amp = fig.add_subplot(gs[0, 0])
# ax_phase = fig.add_subplot(gs[1, 0])
# cax = fig.add_subplot(gs[:, -1])
#
# for _z_0, _alpha_SiC_n, _alpha_Au_n in zip(z_0, alpha_SiC_n.T, alpha_Au_n.T):
#     c = SM.to_rgba(_z_0 * 1e9)
#     ax_amp.plot(wavenumber / 1e2, np.abs(_alpha_SiC_n / _alpha_Au_n), c=c)
#     ax_phase.plot(
#         wavenumber / 1e2, np.unwrap(np.angle(_alpha_SiC_n / _alpha_Au_n), axis=0), c=c
#     )
#
# ax_amp.set(
#     xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
#     xticklabels=[],
#     ylabel=r"$\left|\frac{{\alpha}_{eff, "
#     f"{harmonic}"
#     r", SiC}}{{\alpha}_{eff, "
#     f"{harmonic}"
#     r", Au}}\right|$",
# )
# ax_phase.set(
#     xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
#     xlabel=r"${\omega}$ / cm$^{-1}$",
#     ylabel=r"$\mathrm{arg}\left(\frac{{\alpha}_{eff, "
#     f"{harmonic}"
#     r", SiC}}{{\alpha}_{eff, "
#     f"{harmonic}"
#     r", Au}}\right)$",
# )
#
# fig.colorbar(SM, cax=cax, label=r"$z_0$ / nm")
#
# fig.tight_layout()
# plt.show()
