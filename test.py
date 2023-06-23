import matplotlib.pyplot as plt
import numpy as np

import pysnom


def eps_Lorentz(wavenumber, eps_inf, centre_wavenumber, strength, width):
    """Lorentzian oscillator dielectric function model."""
    return eps_inf + (strength * centre_wavenumber**2) / (
        centre_wavenumber**2 - wavenumber**2 - 1j * width * wavenumber
    )


def eps_Drude(wavenumber, eps_inf, plasma_frequency, gamma):
    """Drude dielectric function model."""
    return eps_inf - (plasma_frequency**2) / (
        wavenumber**2 + 1j * gamma * wavenumber
    )


# Set some experimental parameters
z = 20e-9  # AFM tip height
tapping_amplitude = 20e-9  # AFM tip tapping amplitude
harmonic = np.arange(2, 5)[:, np.newaxis]  # Harmonic for demodulation
wavenumber = np.linspace(1680, 1780, 128) * 1e2

eps_PMMA = eps_Lorentz(wavenumber, 2, 1738e2, 104e-3, 20e2)
beta_PMMA = pysnom.reflection.refl_coeff(1, eps_PMMA)

eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)

# Measurement
alpha_eff_PMMA = pysnom.fdm.eff_pol_n_bulk(
    z=z, tapping_amplitude=tapping_amplitude, harmonic=harmonic, beta=beta_PMMA
)

# Gold reference
alpha_eff_Au = pysnom.fdm.eff_pol_n_bulk(
    z=z, tapping_amplitude=tapping_amplitude, harmonic=harmonic, eps_sample=eps_Au
)

# Normalised complex scattering
sigma_n = alpha_eff_PMMA / alpha_eff_Au

# Recover beta
recovered_beta = pysnom.fdm.refl_coeff_from_eff_pol_n_bulk_Taylor(
    z, tapping_amplitude, harmonic, alpha_eff_PMMA
)

# Plot output
fig, axes = plt.subplots(nrows=4, sharex=True)
n_labels = [str(int(n)) for n in harmonic]

# For neater plotting
k_per_cm = wavenumber * 1e-2

axes[0].plot(k_per_cm, beta_PMMA.real, c="k", ls="-")
axes[0].plot(k_per_cm, beta_PMMA.imag, c="k", ls="--")
axes[0].set_ylabel(r"$\beta$ / a.u.")


axes[1].plot(k_per_cm, np.abs(sigma_n).T, label=n_labels)
axes[1].set_ylabel(r"$s_{n}$ / a.u.")

axes[2].plot(k_per_cm, np.angle(sigma_n).T, label=n_labels)
axes[2].set_ylabel(r"$\phi_{n}$ / radians")

for beta in recovered_beta:
    axes[3].plot(k_per_cm, beta.T.real, "o")
    axes[3].plot(k_per_cm, beta.T.imag, "o")

axes[-1].set(
    xlabel=r"$k$ / cm$^{-1}$",
    xlim=(k_per_cm.max(), k_per_cm.min()),
)

fig.tight_layout()
plt.show()
