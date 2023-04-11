import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import pysnom

# Set some experimental parameters for an oscillating AFM probe
tapping_freq = 2 * np.pi * 250e3  # 250 kHz tapping frequency
tapping_amplitude = 25e-9  # 25 nm oscillation amplitude
z_0 = 10e-9  # 10 nm from sample at bottom of oscillation
periods = 3  # Number of oscillations to show

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Find z as a function of t
t = np.linspace(-periods * np.pi / tapping_freq, periods * np.pi / tapping_freq, 512)
z = z_0 + tapping_amplitude * (1 + np.cos(tapping_freq * t))

# Calculate the effective polarisability
alpha_eff = pysnom.pdm.eff_pol_bulk(z=z, beta=refl_coeff)

# Demodulation
n_max = 5
harmonics = np.arange(n_max + 1)
components = pysnom.pdm.eff_pol_n_bulk(
    z=z_0, tapping_amplitude=tapping_amplitude, harmonic=harmonics, beta=refl_coeff
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
    cmap=cmap,
    norm=BoundaryNorm(boundaries=np.arange(n_max + 2) - 0.5, ncolors=cmap.N),
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
    left=0.145, bottom=0.145, right=0.92, top=0.950, wspace=0.05, hspace=0.25
)

plt.show()