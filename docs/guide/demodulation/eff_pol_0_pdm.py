import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set the height for an AFM approach curve
z_tip = np.linspace(0, 200e-9, 512)

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_env = 1  # Vacuum/air dielectric function
refl_coef_qs = pysnom.reflection.refl_coef_qs(eps_env, eps_Si)

# Calculate the effective polarizability
alpha_eff = pysnom.pdm.eff_pol(z_tip=z_tip, beta=refl_coef_qs)

# Plot output
fig, ax = plt.subplots()
z_nm = z_tip * 1e9  # For neater plotting
ax.plot(z_nm, alpha_eff.real)
ax.set(
    xlabel=r"$z_{tip}$ / nm",
    ylabel=r"$\Re(\alpha_{eff})$",
    xlim=(z_nm.min(), z_nm.max()),
)
fig.tight_layout()
plt.show()
