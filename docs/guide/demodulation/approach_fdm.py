import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an AFM approach curve
z_tip = np.linspace(0, 60e-9, 512)  # Define an approach curve
A_tip = 20e-9  # AFM tip tapping amplitude
harmonics = np.array([2, 3, 4])  # Harmonics for demodulation

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Calculate the effective polarisability using demod
# offset by tapping amplitude so oscillation doesn't intersect with sample
z_shift = z_tip + A_tip
alpha_eff_demod = pysnom.demodulate.demod(
    f_x=pysnom.fdm.eff_pol_bulk,
    x_0=z_shift[:, np.newaxis],  # newaxis added for array broadcasting
    x_amplitude=A_tip,
    harmonic=harmonics,
    f_args=(refl_coeff,),
)

# Calculate the effective polarisability directly
alpha_eff_direct = pysnom.fdm.eff_pol_n_bulk(
    z_tip=z_tip[:, np.newaxis],  # newaxis added for array broadcasting
    A_tip=A_tip,
    harmonic=harmonics,
    beta=refl_coeff,
)

# Normalize to value at z_tip = 0
alpha_eff_demod /= alpha_eff_demod[0]
alpha_eff_direct /= alpha_eff_direct[0]

# Plot output
fig, ax = plt.subplots()
z_nm = z_tip * 1e9  # For neater plotting
ax.plot(
    z_nm,
    np.real(alpha_eff_demod),
    label=[f"demod(eff_pol_0): $n = ${n}" for n in harmonics],
)
ax.plot(
    z_nm,
    np.real(alpha_eff_direct),
    ls="--",
    label=[f"eff_pol: $n = ${n}" for n in harmonics],
)
ax.set(
    xlabel=r"$z_{tip}$ / nm",
    ylabel=r"$\Re\left(\alpha_{eff, n} / \alpha_{eff, n, z_{tip}=0}\right)$",
    xlim=(z_nm.min(), z_nm.max()),
)
ax.legend(ncol=2)
fig.tight_layout()
plt.show()
