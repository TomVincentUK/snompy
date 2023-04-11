import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import pysnom


def eps_Lorentz(wavenumber, eps_inf, centre_wavenumber, strength, width):
    """Lorentzian oscillator dielectric function model."""
    return eps_inf + (strength * centre_wavenumber**2) / (
        centre_wavenumber**2 - wavenumber**2 - 1j * width * wavenumber
    )


def eps_Drude(wavenumber, eps_inf, plasma_frequency, gamma):
    """Drude dielectric function model."""
    return eps_inf - (plasma_frequency**2) / (
        wavenumber**2 + 1j * gamma * wavenumber
    )


# Set some experimental parameters
z = 20e-9  # AFM tip height
tapping_amplitude = 20e-9  # AFM tip tapping amplitude
radius = 30e-9  # AFM tip radius of curvature
semi_maj_axis = 350e-9  # Semi-major axis length of ellipsoid tip model
harmonic = 3  # Harmonic for demodulation
wavenumber = np.linspace(1680, 1780, 128) * 1e2

# Semi-infinite superstrate and substrate
eps_air = 1
eps_Si = 11.7  # Si dielectric function in the mid-infrared

# Very simplified model of PMMA dielectric function based on ref [1] below
eps_PMMA = eps_Lorentz(wavenumber, 2, 1738e2, 14e-3, 20e2)
PMMA_thickness = np.geomspace(10, 100, 32) * 1e-9

# Model of Au dielectric function from ref [2] below
eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)

# Measurement
alpha_eff_PMMA = pysnom.fdm.eff_pol_n_multi(
    z=z,
    tapping_amplitude=tapping_amplitude,
    harmonic=harmonic,
    eps_stack=(eps_air, eps_PMMA, eps_Si),
    t_stack=(PMMA_thickness[:, np.newaxis],),
    radius=radius,
    semi_maj_axis=semi_maj_axis,
)

# Gold reference
alpha_eff_Au = pysnom.fdm.eff_pol_n_bulk(
    z=z,
    tapping_amplitude=tapping_amplitude,
    harmonic=harmonic,
    eps_sample=eps_Au,
    eps_environment=eps_air,
    radius=radius,
    semi_maj_axis=semi_maj_axis,
)

# Normalised complex scattering
sigma_n = alpha_eff_PMMA / alpha_eff_Au

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)

# For neater plotting
k_per_cm = wavenumber * 1e-2
thickness_nm = PMMA_thickness * 1e9

SM = plt.cm.ScalarMappable(
    cmap=plt.cm.Spectral_r,
    norm=Normalize(vmin=thickness_nm.min(), vmax=thickness_nm.max()),
)  # This maps thickness to colour

for t, sigma in zip(thickness_nm, sigma_n):
    c = SM.to_rgba(t)
    axes[0].plot(k_per_cm, np.abs(sigma), c=c)
    axes[1].plot(k_per_cm, np.angle(sigma), c=c)

axes[0].set_ylabel(r"$s_{" f"{harmonic}" r"}$ / a.u.")
axes[1].set(
    xlabel=r"$k$ / cm$^{-1}$",
    ylabel=r"$\phi_{" f"{harmonic}" r"}$ / radians",
    xlim=(k_per_cm.max(), k_per_cm.min()),
)
fig.tight_layout()
cbar = fig.colorbar(SM, ax=axes, label="PMMA thickness / nm")
plt.show()
