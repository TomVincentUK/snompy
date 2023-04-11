import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Define an approach curve on Si
z_nm = np.linspace(0, 100, 512)  # Useful for plotting
z = z_nm * 1e-9  # Convert to nm to m (we'll work in SI units)
tapping_amplitude = 25e-9
eps_sample = 11.7  # The mid-IR dielectric function of Si

# Set up an axis for plotting
fig, ax = plt.subplots()
ax.set(
    xlabel=r"$z$",
    xlim=(z_nm.min(), z_nm.max()),
    ylabel=r"$\frac{\alpha_{eff, \ n}}{(\alpha_{eff, \ n})|_{z = 0}}$",
)
fig.tight_layout()

# Calculate an approach curve using default parameters
single_harmonic = 2
alpha_eff = pysnom.fdm.eff_pol_n_bulk(
    z=z,
    tapping_amplitude=tapping_amplitude,
    harmonic=single_harmonic,
    eps_sample=eps_sample,
)
alpha_eff /= alpha_eff[0]  # Normalise to z=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Default parameters ($\varepsilon$), $n = " f"{single_harmonic}" r"$",
)
ax.legend()

# Use beta instead of eps_sample
beta = pysnom.reflection.refl_coeff(1, eps_sample)
alpha_eff = pysnom.fdm.eff_pol_n_bulk(
    z=z,
    tapping_amplitude=tapping_amplitude,
    harmonic=single_harmonic,
    beta=beta,
)
alpha_eff /= alpha_eff[0]  # Normalise to z=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Default parameters ($\beta$), $n = " f"{single_harmonic}" r"$",
    ls="--",
)
ax.legend()  # Update the legend

# Change the default parameters
radius = 100e-9
semi_maj_axis = 400e-9
g_factor = 0.7
alpha_eff = pysnom.fdm.eff_pol_n_bulk(
    z=z,
    tapping_amplitude=tapping_amplitude,
    harmonic=single_harmonic,
    eps_sample=eps_sample,
    radius=radius,
    semi_maj_axis=semi_maj_axis,
    g_factor=g_factor,
)
alpha_eff /= alpha_eff[0]  # Normalise to z=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Custom parameters, $n = " f"{single_harmonic}" r"$",
    ls=":",
)
ax.legend()  # Update the legend

# Vector broadcasting
multiple_harmonics = np.arange(3, 6)
alpha_eff = pysnom.fdm.eff_pol_n_bulk(
    z=z[:, np.newaxis],  # newaxis added for array broadcasting
    tapping_amplitude=tapping_amplitude,
    harmonic=multiple_harmonics,
    eps_sample=eps_sample,
    radius=radius,
    semi_maj_axis=semi_maj_axis,
    g_factor=g_factor,
)
alpha_eff /= alpha_eff[0]  # Normalise to z=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=[r"Custom parameters, $n = " f"{n}" r"$" for n in multiple_harmonics],
    ls=":",
)
ax.legend()  # Update the legend

fig.show()
