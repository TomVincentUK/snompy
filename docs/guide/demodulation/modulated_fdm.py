import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an oscillating AFM probe
tapping_freq = 2 * np.pi * 250e3  # 250 kHz tapping frequency
A_tip = 25e-9  # 25 nm oscillation amplitude
z_bottom = 10e-9  # 10 nm from sample at bottom of oscillation
periods = 3  # Number of oscillations to show

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_env = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_env, eps_Si)

# Find z_tip as a function of t
t = np.linspace(-periods * np.pi / tapping_freq, periods * np.pi / tapping_freq, 512)
z_tip = z_bottom + A_tip * (1 + np.cos(tapping_freq * t))

# Calculate the effective polarisability
alpha_eff = pysnom.fdm.bulk.eff_pol(z_tip=z_tip, beta=refl_coeff)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
t_us = t * 1e6  # For neater plotting
z_nm = z_tip * 1e9  # For neater plotting
axes[0].plot(t_us, z_nm)
axes[1].plot(t_us, alpha_eff.real)
axes[0].set(ylabel=r"$z_{tip}$ / nm", ylim=(0, None))
axes[1].set(
    xlabel=r"$t$ / $\mathrm{\mu}$s",
    ylabel=r"$\Re(\alpha_{eff})$",
    xlim=(t_us.min(), t_us.max()),
)
fig.tight_layout()
plt.show()
