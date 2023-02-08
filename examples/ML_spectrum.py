"""
Simulated PMMA on Si spectra with different thickness for PMMA.

References
----------
.. [1] Z. M. Zhang, G. Lefever-Button, F. R. Powell,
   Infrared refractive index and extinction coefficient of polyimide films,
   Int. J. Thermophys., 19 (1998) 905.
   https://doi.org/10.1023/A:1022655309574.
.. [2] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W.
   Alexander, C.A. Ward,
   Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
   Ti, and W in the infrared and far infrared,
   Appl. Opt. 22 (1983) 1099.
   https://doi.org/10.1364/AO.22.001099.
.. [3] Lars Mester Nat. Comms. (2020).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import pysnom as fdm


def eps_Lorentz(omega, eps_inf, omega_0, strength, gamma):
    """
    Lorentzian oscillator dielectric function model. Function definition
    from equation (5) of reference [1]_.
    """
    return eps_inf + (strength * omega_0**2) / (
        omega_0**2 - omega**2 - 1j * gamma * omega
    )


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (2)
    of reference [2]_.
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = np.linspace(1680, 1780, 128) * 1e2
z_0 = 50e-9
tapping_amplitude = 50e-9
radius = 20e-9
harmonic = 3

# Constant dielectric functions
# Values from [3]_
eps_air = 1
eps_Si = 11.7

# Dispersive dielectric functions
# (My simplified model for the PMMA C=O bond based on fig 5a of [3]_)
eps_PMMA = eps_Lorentz(wavenumber, 2, 1738e2, 14e-3, 20e2)
eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]_

eps_stack = eps_air, eps_PMMA, eps_Si
eps_stack_ref = eps_air, eps_Au

t_PMMA = np.linspace(0, 100, 10)[:, np.newaxis] * 1e-9

alpha_eff = fdm.multilayer.eff_pol_ML(
    z_0, tapping_amplitude, harmonic, eps_stack=eps_stack, t_stack=(t_PMMA,)
)
alpha_eff_ref = fdm.multilayer.eff_pol_ML(
    z_0, tapping_amplitude, harmonic, eps_stack=eps_stack_ref
)
signal = alpha_eff / alpha_eff_ref

# Plotting
fig = plt.figure()
gs = plt.GridSpec(nrows=3, ncols=2, width_ratios=(1, 0.1))
SM = plt.cm.ScalarMappable(
    cmap=plt.cm.Spectral,
    norm=Normalize(vmin=t_PMMA.min() * 1e9, vmax=t_PMMA.max() * 1e9),
)
ax_eps = fig.add_subplot(gs[0, 0])
ax_eps.plot(wavenumber / 1e2, eps_PMMA.real, label="real")
ax_eps.plot(wavenumber / 1e2, eps_PMMA.imag, label="imaginary")

ax_amp = fig.add_subplot(gs[1, 0])
ax_phase = fig.add_subplot(gs[2, 0])
cax = fig.add_subplot(fig.add_subplot(gs[1:, -1]))
for t, s in zip(t_PMMA, signal):
    c = SM.to_rgba(t * 1e9)
    ax_amp.plot(wavenumber / 1e2, np.abs(s), c=c)
    ax_phase.plot(wavenumber / 1e2, np.unwrap(np.angle(s), axis=0), c=c)

ax_eps.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xticklabels=[],
    ylabel=r"$\varepsilon_{PMMA}$",
)
ax_eps.legend()
ax_amp.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xticklabels=[],
    ylabel=r"$\left|\frac{{\alpha}_{eff, "
    f"{harmonic}"
    r", PMMA-Si}}{{\alpha}_{eff, "
    f"{harmonic}"
    r", Au}}\right|$",
)
ax_phase.set(
    xlim=wavenumber[0 :: wavenumber.size - 1] * 1e-2,
    xlabel=r"${\omega}$ / cm$^{-1}$",
    ylabel=r"$\mathrm{arg}\left(\frac{{\alpha}_{eff, "
    f"{harmonic}"
    r", PMMA-Si}}{{\alpha}_{eff, "
    f"{harmonic}"
    r", Au}}\right)$",
)

fig.colorbar(SM, cax=cax, label=r"$t_{PMMA}$ / nm")

fig.tight_layout()
plt.show()
