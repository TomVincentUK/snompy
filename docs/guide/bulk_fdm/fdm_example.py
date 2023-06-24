import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Define an approach curve on Si
z_nm = np.linspace(0, 100, 512)  # Useful for plotting
z_tip = z_nm * 1e-9  # Convert to nm to m (we'll work in SI units)
A_tip = 25e-9
eps_samp = 11.7  # The mid-IR dielectric function of Si

# Set up an axis for plotting
fig, ax = plt.subplots()
ax.set(
    xlabel=r"$z_{tip}$",
    xlim=(z_nm.min(), z_nm.max()),
    ylabel=r"$\frac{\alpha_{eff, \ n}}{(\alpha_{eff, \ n})|_{z_{tip} = 0}}$",
)
fig.tight_layout()

# Calculate an approach curve using default parameters
single_harmonic = 2
alpha_eff = pysnom.fdm.bulk.eff_pol_n(
    z_tip=z_tip,
    A_tip=A_tip,
    n=single_harmonic,
    eps_samp=eps_samp,
)
alpha_eff /= alpha_eff[0]  # Normalise to z_tip=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Default parameters ($\varepsilon$), $n = " f"{single_harmonic}" r"$",
)
ax.legend()

# Use beta instead of eps_samp
beta = pysnom.reflection.refl_coeff(1, eps_samp)
alpha_eff = pysnom.fdm.bulk.eff_pol_n(
    z_tip=z_tip,
    A_tip=A_tip,
    n=single_harmonic,
    beta=beta,
)
alpha_eff /= alpha_eff[0]  # Normalise to z_tip=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Default parameters ($\beta$), $n = " f"{single_harmonic}" r"$",
    ls="--",
)
ax.legend()  # Update the legend

# Change the default parameters
r_tip = 100e-9
L_tip = 400e-9
g_factor = 0.7
alpha_eff = pysnom.fdm.bulk.eff_pol_n(
    z_tip=z_tip,
    A_tip=A_tip,
    n=single_harmonic,
    eps_samp=eps_samp,
    r_tip=r_tip,
    L_tip=L_tip,
    g_factor=g_factor,
)
alpha_eff /= alpha_eff[0]  # Normalise to z_tip=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=r"Custom parameters, $n = " f"{single_harmonic}" r"$",
    ls=":",
)
ax.legend()  # Update the legend

# Vector broadcasting
multiple_harmonics = np.arange(3, 6)
alpha_eff = pysnom.fdm.bulk.eff_pol_n(
    z_tip=z_tip[:, np.newaxis],  # newaxis added for array broadcasting
    A_tip=A_tip,
    n=multiple_harmonics,
    eps_samp=eps_samp,
    r_tip=r_tip,
    L_tip=L_tip,
    g_factor=g_factor,
)
alpha_eff /= alpha_eff[0]  # Normalise to z_tip=0
ax.plot(
    z_nm,
    np.abs(alpha_eff),
    label=[r"Custom parameters, $n = " f"{n}" r"$" for n in multiple_harmonics],
    ls=":",
)
ax.legend()  # Update the legend

fig.show()
