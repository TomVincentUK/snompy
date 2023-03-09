import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an oscillating AFM probe
harmonic = 3
tapping_amplitude = 25e-9  # 25 nm oscillation amplitude
z_0 = 10e-9  # 10 nm from sample at bottom of oscillation

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Find z as a function of theta
theta = np.linspace(-np.pi, np.pi, 512)
z = z_0 + tapping_amplitude * (1 + np.cos(theta))

# Calculate the effective polarisability
alpha_eff = pysnom.fdm.eff_pol_0_bulk(z=z, beta=refl_coeff)

# Generate a complex sinusoidal envelope
envelope = np.exp(1j * harmonic * theta)

# Get the integrand and evaluate the integral using the trapezium method
integrand = alpha_eff * envelope
alpha_eff_n = np.trapz(integrand, theta)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
for ax, component in zip(axes, (np.real, np.imag)):
    ax.plot(theta, component(alpha_eff), label=r"$\alpha_{eff}(\theta)$")
    ax.plot(
        theta,
        component(envelope),
        c="k",
        ls="--",
        label=r"$e^{" f"{harmonic}" r"i \theta}$",
    )
    ax.fill_between(theta, 0, component(integrand), alpha=0.3, label="integrand")
axes[0].set(ylabel=r"$\Re(\alpha_{eff})$")
axes[-1].set(
    xlabel=r"$\theta$",
    ylabel=r"$\Im(\alpha_{eff})$",
    xlim=(theta.min(), theta.max()),
)
axes[0].legend()
axes[0].set_title(  # Print integration result as title
    r"$\alpha_{eff, "
    f"{harmonic}"
    r"} = \int_{-\pi}^{\pi} \alpha_{eff}(\theta) e^{"
    f"{harmonic}"
    r"i \theta} d\theta \approx"
    f"{np.abs(alpha_eff_n):.2f}"
    r"e^{"
    f"{np.angle(alpha_eff_n):.2f}"
    r"i}$"
)
fig.tight_layout()
plt.show()