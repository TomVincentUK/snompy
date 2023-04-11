import matplotlib.pyplot as plt
import numpy as np

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
alpha_eff = pysnom.fdm.eff_pol_bulk(z=z, beta=refl_coeff)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
t_us = t * 1e6  # For neater plotting
z_nm = z * 1e9  # For neater plotting
axes[0].plot(t_us, z_nm)
axes[1].plot(t_us, alpha_eff.real)
axes[0].set(ylabel=r"$z$ / nm", ylim=(0, None))
axes[1].set(
    xlabel=r"$t$ / $\mathrm{\mu}$s",
    ylabel=r"$\Re(\alpha_{eff})$",
    xlim=(t_us.min(), t_us.max()),
)
fig.tight_layout()
plt.show()
