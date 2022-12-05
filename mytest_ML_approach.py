import matplotlib.pyplot as plt
import numpy as np

import finite_dipole as fdm

# Values from Lars Mester Nat. Comms. (2020)
eps_air = 1
eps_PS = 2.5
eps_PMMA = 1.52 + 0.83j
eps_Si = 11.7
eps_stack = eps_air, eps_PS, eps_PMMA, eps_Si

t_PS = 100e-9
t_PMMA = 100e-9
t_stack = (t_PS, t_PMMA)

beta_stack = fdm.tools.refl_coeff(eps_stack[:-1], eps_stack[1:])

N = 5
z_0 = np.linspace(0, 35, N)[..., np.newaxis] * 1e-9
tapping_amplitude = 20e-9
harmonic = np.arange(1, 5, 1)

alpha_eff = fdm.multilayer.eff_pol_ML(
    z_0, tapping_amplitude, harmonic, beta_stack=beta_stack, t_stack=t_stack
)

# Normalize to z = 0
alpha_eff /= alpha_eff[0]

# Plotting
fig, (ax_amp, ax_phase) = plt.subplots(nrows=2, sharex=True)

linestyles = "-", "--", "-.", ":"
for n, _alpha_eff, ls in zip(harmonic, alpha_eff.T, linestyles):
    ax_amp.plot(z_0 * 1e9, np.abs(_alpha_eff), ls=ls, label=r"$n=" f"{n}" "$")
    ax_phase.plot(z_0 * 1e9, np.unwrap(np.angle(_alpha_eff), axis=0), ls=ls)

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
