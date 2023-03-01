import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an AFM approach curve
z = np.linspace(0, 200e-9, 512)  # Define an approach curve
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Calculate the effective polarisability
alpha_eff = pysnom.fdm.eff_pol_0_bulk(z=z, beta=refl_coeff)

# Plot output
fig, ax = plt.subplots()
z_nm = z * 1e9  # For neater plotting
ax.plot(z_nm, np.abs(alpha_eff))
ax.set(
    xlabel=r"$z$ / nm",
    ylabel=r"$|\alpha_{eff}|$",
    xlim=(z_nm.min(), z_nm.max()),
)
fig.tight_layout()
plt.show()
