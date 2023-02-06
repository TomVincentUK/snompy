import matplotlib.pyplot as plt
import numpy as np

import finite_dipole as fdm

# Set some experimental parameters for an AFM approach curve
z = np.linspace(0, 100e-9, 512)  # Define an approach curve
tapping_amplitude = 20e-9  # AFM tip tapping amplitude
harmonics = np.array([2, 3, 4])  # Harmonics for demodulation
eps_Si = 11.7  # Si dielectric function in the mid-infrared

# Calculate the effective polarisability
alpha_eff = fdm.bulk.eff_pol(
    z=z[:, np.newaxis],  # newaxis added for array broadcasting
    tapping_amplitude=tapping_amplitude,
    harmonic=harmonics,
    eps_sample=eps_Si,
)

# Normalize to value at z = 0
alpha_eff /= alpha_eff[0]

# Plot output
fig, ax = plt.subplots()
z_nm = z * 1e9  # For neater plotting
ax.plot(z_nm, np.abs(alpha_eff), label=[f"$n = ${n}" for n in harmonics])
ax.set(
    xlabel=r"$z$ / nm",
    ylabel=r"$\alpha_{eff, n} / \alpha_{eff, n, z=0}$",
    xlim=(z_nm.min(), z_nm.max()),
)
ax.legend()
fig.tight_layout()
plt.show()