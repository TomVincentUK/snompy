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
z_tip = 0  # AFM tip height
A_tip = 20e-9  # AFM tip tapping amplitude
r_tip = 30e-9  # AFM tip radius of curvature
L_tip = 350e-9  # Semi-major axis length of ellipsoid tip model
n = 3  # Harmonic for demodulation
theta_in = np.deg2rad(60)  # Light angle of incidence
c_r = 0.3  # Experimental weighting factor
k_vac = np.linspace(1680, 1800, 128) * 1e2  # Vacuum wavenumber
method = "Mester"  # The FDM method to use

# Semi-infinite superstrate and substrate
eps_air = 1
eps_Si = 11.7  # Si permitivitty in the mid-infrared

# Very simplified model of PMMA dielectric function based on ref [1] below
eps_pmma = eps_Lorentz(k_vac, 2, 1738e2, 14e-3, 20e2)
t_pmma = np.geomspace(1, 35, 32) * 1e-9  # A range of thicknesses
sample_pmma = pysnom.Sample(
    eps_stack=(eps_air, eps_pmma, eps_Si),
    t_stack=(t_pmma[:, np.newaxis],),
    k_vac=k_vac,
)

# Model of Au dielectric function from ref [2] below
eps_Au = eps_Drude(k_vac, 1, 7.25e6, 2.16e4)
sample_Au = pysnom.bulk_sample(eps_sub=eps_Au, eps_env=eps_air, k_vac=k_vac)

# Measurement
alpha_eff_pmma = pysnom.fdm.eff_pol_n(
    z_tip=z_tip,
    A_tip=A_tip,
    n=n,
    sample=sample_pmma,
    r_tip=r_tip,
    L_tip=L_tip,
    method=method,
)
r_coef_pmma = sample_pmma.refl_coef(theta_in=theta_in)
sigma_pmma = (1 + c_r * r_coef_pmma) ** 2 * alpha_eff_pmma

# Gold reference
alpha_eff_Au = pysnom.fdm.eff_pol_n(
    z_tip=z_tip,
    A_tip=A_tip,
    n=n,
    sample=sample_Au,
    r_tip=r_tip,
    L_tip=L_tip,
    method=method,
)
r_coef_Au = sample_Au.refl_coef(theta_in=theta_in)
sigma_Au = (1 + c_r * r_coef_Au) ** 2 * alpha_eff_Au

# Normalised complex scattering
sigma_n = sigma_pmma / sigma_Au

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)

# For neater plotting
k_per_cm = k_vac * 1e-2
t_nm = t_pmma * 1e9

SM = plt.cm.ScalarMappable(
    cmap=plt.cm.Spectral_r,
    norm=Normalize(vmin=t_nm.min(), vmax=t_nm.max()),
)  # This maps thickness to colour

for t, sigma in zip(t_nm, sigma_n):
    c = SM.to_rgba(t)
    axes[0].plot(k_per_cm, np.abs(sigma), c=c)
    axes[1].plot(k_per_cm, np.angle(sigma), c=c)

axes[0].set_ylabel(r"$s_{" f"{n}" r"}$ / a.u.")
axes[1].set(
    xlabel=r"$k$ / cm$^{-1}$",
    ylabel=r"$\phi_{" f"{n}" r"}$ / radians",
    xlim=(k_per_cm.max(), k_per_cm.min()),
)
fig.tight_layout()
cbar = fig.colorbar(SM, ax=axes, label="PMMA thickness / nm")
plt.show(block=False)
