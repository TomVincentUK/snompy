import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import snompy

# Set some experimental parameters for an oscillating AFM probe
tapping_freq = 2 * np.pi * 250e3  # 250 kHz tapping frequency
A_tip = 25e-9  # 25 nm oscillation amplitude
z_bottom = 10e-9  # 10 nm from sample at bottom of oscillation
periods = 3  # Number of oscillations to show

# Material parameters
eps_Si = 11.7  # Si permitivitty in the mid-infrared
eps_env = 1  # Vacuum/air permitivitty
refl_coef_qs = snompy.reflection.refl_coef_qs(eps_env, eps_Si)

# Find z_tip as a function of t
t = np.linspace(-periods * np.pi / tapping_freq, periods * np.pi / tapping_freq, 512)
z_tip = z_bottom + A_tip * (1 + np.cos(tapping_freq * t))

# Calculate the effective polarizability
alpha_eff = snompy.fdm.bulk.eff_pol(z_tip=z_tip, beta=refl_coef_qs)

# Demodulation
n_max = 5
harmonics = np.arange(n_max + 1)
components = snompy.fdm.bulk.eff_pol_n(
    z_tip=z_bottom, A_tip=A_tip, n=harmonics, beta=refl_coef_qs
)
waves = [components[n] * np.exp(1j * t * tapping_freq * n) for n in harmonics]

# Plot output
fig = plt.figure()
gs = plt.GridSpec(
    figure=fig,
    nrows=2,
    ncols=2,
    width_ratios=(1, 0.05),
    height_ratios=(np.ptp(alpha_eff.real), np.ptp(np.real(waves[1:]))),
)
axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
t_us = t * 1e6  # For neater plotting

# Plot total
axes[0].plot(t_us, alpha_eff.real, c="k", label="total signal")

# Plot waves
cmap = plt.cm.Spectral
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=BoundaryNorm(boundaries=np.arange(n_max + 2) - 0.5, ncolors=cmap.N)
)
axes[0].plot(t_us, waves[0].real, ls="--", c=sm.to_rgba(0), label="components")
for i, wave in enumerate(waves[1:]):
    axes[1].plot(t_us, wave.real, ls="--", c=sm.to_rgba(i + 1), zorder=-i)


axes[0].legend()
axes[0].spines["bottom"].set_visible(False)
axes[0].set(xlim=(t_us.min(), t_us.max()), xticks=[])
axes[1].set(xlabel=r"$t$ / $\mathrm{\mu}$s", xlim=(t_us.min(), t_us.max()))
axes[0].set_ylabel(r"$\Re(\alpha_{eff})$", y=0.2)

cax = fig.add_subplot(gs[:, -1])
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(n_max + 1), label="$n$", extend="max")
cbar.outline.set_visible(False)

fig.tight_layout()

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
axes[0].plot([0], [0], transform=axes[0].transAxes, **kwargs)
axes[1].plot([0], [1], transform=axes[1].transAxes, **kwargs)

fig.subplots_adjust(
    left=0.145, bottom=0.145, right=0.92, top=0.985, wspace=0.05, hspace=0.05
)

plt.show()
