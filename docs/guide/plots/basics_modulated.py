import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an AFM approach curve
tapping_freq = 250e3
tapping_amplitude = 25e-9
z_0 = 10e-9
periods = 3
t = np.linspace(-periods * np.pi / tapping_freq, periods * np.pi / tapping_freq, 512)
z = z_0 + tapping_amplitude * (1 + np.cos(tapping_freq * t))

eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Calculate the effective polarisability
alpha_eff = pysnom.fdm.eff_pol_0_bulk(z=z, beta=refl_coeff)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
t_us = t * 1e6  # For neater plotting
z_nm = z * 1e9  # For neater plotting
axes[0].plot(t_us, z_nm)
axes[0].set(ylabel=r"$z$ / nm", ylim=(0, None))
axes[1].plot(t_us, np.abs(alpha_eff))
axes[1].set(
    xlabel=r"$t$ / $\mathrm{\mu}$s",
    ylabel=r"$|\alpha_{eff}|$",
    xlim=(t_us.min(), t_us.max()),
)
fig.tight_layout()
plt.show()
