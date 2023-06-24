import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an AFM approach curve
z_tip = np.linspace(0, 60e-9, 512)  # Define an approach curve
A_tip = 20e-9  # AFM tip tapping amplitude
harmonics = np.array([2, 3, 4])  # Harmonics for demodulation
eps_Si = 11.7  # Si dielectric function in the mid-infrared

# Calculate the effective polarisability using FDM and PDM
alpha_eff_fdm = pysnom.fdm.bulk.eff_pol_n(
    z_tip=z_tip[:, np.newaxis],  # newaxis added for array broadcasting
    A_tip=A_tip,
    n=harmonics,
    eps_samp=eps_Si,
)
alpha_eff_pdm = pysnom.pdm.eff_pol_n(
    z_tip=z_tip[:, np.newaxis],  # newaxis added for array broadcasting
    A_tip=A_tip,
    n=harmonics,
    eps_samp=eps_Si,
)

# Normalize to value at z_tip = 0
alpha_eff_fdm /= alpha_eff_fdm[0]
alpha_eff_pdm /= alpha_eff_pdm[0]

# Plot output
fig, ax = plt.subplots()
z_nm = z_tip * 1e9  # For neater plotting
ax.plot(z_nm, np.abs(alpha_eff_fdm), label=[f"FDM: $n = ${n}" for n in harmonics])
ax.plot(
    z_nm, np.abs(alpha_eff_pdm), label=[f"PDM: $n = ${n}" for n in harmonics], ls="--"
)
ax.set(
    xlabel=r"$z_{tip}$ / nm",
    ylabel=r"$\left|\alpha_{eff, n} / \alpha_{eff, n, z_{tip}=0}\right|$",
    xlim=(z_nm.min(), z_nm.max()),
)
ax.legend(ncol=2)
fig.tight_layout()
plt.show()
