"""
Test of finite dipole model (FDM) by reproducing figure 7(a) of reference [1].
They didn't specify the wavenumber, but the results seem quite insensitive to
it, particularly after normalisation. This script also shows results for
multiple harmonics.

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

from finite_dipole import eff_polarizability_nth


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (2) of
    reference [2].
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = 1000 * 1e2
z_0 = np.linspace(0, 35, 512)[..., np.newaxis] * 1e-9
tapping_amplitude = 18e-9
radius = 20e-9
harmonic = np.arange(1, 5, 1)

eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]
alpha_Au_n = eff_polarizability_nth(
    z_0, tapping_amplitude, harmonic, eps_sample=eps_Au, radius=radius
)

# Normalize to z = 0
alpha_Au_n *= np.exp(-1j * np.angle(alpha_Au_n[0])) / np.abs(alpha_Au_n[0])

# Plotting
fig, (ax_amp, ax_phase) = plt.subplots(nrows=2, sharex=True)

linestyles = "-", "--", "-.", ":"
for n, _alpha_Au_n, ls in zip(harmonic, alpha_Au_n.T, linestyles):
    ax_amp.plot(z_0 * 1e9, np.abs(_alpha_Au_n), ls=ls, label=r"$n=" f"{n}" "$")
    ax_phase.plot(z_0 * 1e9, np.unwrap(np.angle(_alpha_Au_n), axis=0), ls=ls)

ax_amp.set(
    ylabel=r"$\left|{\alpha}_{eff, n, Au}\right|$",
)
ax_amp.legend()
ax_phase.set(
    xlim=z_0[0 :: z_0.size - 1] * 1e9,
    xlabel=r"$z_0$ / nm",
    ylabel=r"$\mathrm{arg}\left({\alpha}_{eff, n, Au}\right)$",
)

fig.tight_layout()
plt.show()
