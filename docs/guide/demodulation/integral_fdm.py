import matplotlib.pyplot as plt
import numpy as np

import snompy

# Sent some experimental parameters for an oscillating AFM probe
n = 3
A_tip = 25e-9  # 25 nm oscillation amplitude
z_bottom = 10e-9  # 10 nm from sample at bottom of oscillation

# Material parameters
eps_Si = 11.7  # Si permitivitty in the mid-infrared
eps_env = 1  # Vacuum/air permitivitty
refl_coef_qs = snompy.reflection.refl_coef_qs(eps_env, eps_Si)

# Find z_tip as a function of theta
theta = np.linspace(-np.pi, np.pi, 512)
z_tip = z_bottom + A_tip * (1 + np.cos(theta))

# Calculate the effective polarizability
alpha_eff = snompy.fdm.bulk.eff_pol(z_tip=z_tip, beta=refl_coef_qs)

# Generate a complex sinusoidal envelope
envelope = np.exp(1j * n * theta)

# Get the integrand and evaluate the integral using the trapezium method
integrand = alpha_eff * envelope
alpha_eff_n = np.trapz(integrand, theta)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
for ax, component in zip(axes, (np.real, np.imag)):
    ax.plot(theta, component(alpha_eff), label=r"$\alpha_{eff}(\theta)$")
    ax.plot(
        theta, component(envelope), c="k", ls="--", label=r"$e^{" f"{n}" r"i \theta}$"
    )
    ax.fill_between(theta, 0, component(integrand), alpha=0.3, label="integrand")
axes[0].set(ylabel=r"$\Re(\alpha_{eff})$")
axes[-1].set(
    xlabel=r"$\theta$", ylabel=r"$\Im(\alpha_{eff})$", xlim=(theta.min(), theta.max())
)
axes[0].legend()
axes[0].set_title(  # Print integration result as title
    r"$\alpha_{eff, "
    f"{n}"
    r"} = \int_{-\pi}^{\pi} \alpha_{eff}(\theta) e^{"
    f"{n}"
    r"i \theta} d\theta \approx"
    f"{np.abs(alpha_eff_n):.2f}"
    r"e^{"
    f"{np.angle(alpha_eff_n):.2f}"
    r"i}$"
)
fig.tight_layout()
plt.show()
