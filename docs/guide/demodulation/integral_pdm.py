import matplotlib.pyplot as plt
import numpy as np

import pysnom

# Set some experimental parameters for an oscillating AFM probe
harmonic = 3
A_tip = 25e-9  # 25 nm oscillation amplitude
z_bottom = 10e-9  # 10 nm from sample at bottom of oscillation

# Material parameters
eps_Si = 11.7  # Si dielectric function in the mid-infrared
eps_environment = 1  # Vacuum/air dielectric function
refl_coeff = pysnom.reflection.refl_coeff(eps_environment, eps_Si)

# Find z_tip as a function of theta
theta = np.linspace(-np.pi, np.pi, 512)
z_tip = z_bottom + A_tip * (1 + np.cos(theta))

# Calculate the effective polarisability
alpha_eff = pysnom.pdm.eff_pol_bulk(z_tip=z_tip, beta=refl_coeff)

# Generate a complex sinusoidal envelope
envelope = np.exp(1j * harmonic * theta)

# Get the integrand and evaluate the integral using the trapezium method
integrand = alpha_eff * envelope
alpha_eff_n = np.trapz(integrand, theta)

# Plot output
fig, axes = plt.subplots(nrows=2, sharex=True)
twins = [ax.twinx() for ax in axes]  # Twin axes because |alpha_eff| << 1
for ax, twin, component in zip(axes, twins, (np.real, np.imag)):
    (f,) = ax.plot(theta, component(alpha_eff), label=r"$\alpha_{eff}(\theta)$")
    filled = ax.fill_between(
        theta, 0, component(integrand), alpha=0.3, label="integrand"
    )

    # Twin plots
    (env,) = twin.plot(
        theta,
        component(envelope),
        c="k",
        ls="--",
        label=r"$e^{" f"{harmonic}" r"i \theta}$",
    )
    for side, visible in zip(
        ("top", "right", "bottom", "left"), (False, True, False, False)
    ):
        twin.spines[side].set_visible(visible)
axes[0].set(ylabel=r"$\Re(\alpha_{eff})$")
axes[-1].set(
    xlabel=r"$\theta$",
    ylabel=r"$\Im(\alpha_{eff})$",
    xlim=(theta.min(), theta.max()),
)
axes[0].legend(handles=(f, env, filled))
axes[0].set_title(  # Print integration result as title
    r"$\alpha_{eff, "
    f"{harmonic}"
    r"} = \int_{-\pi}^{\pi} \alpha_{eff}(\theta) e^{"
    f"{harmonic}"
    r"i \theta} d\theta \approx ("
    f"{np.abs(alpha_eff_n):.2e}"
    r") e^{"
    f"{np.angle(alpha_eff_n):.2f}"
    r"i}$"
)
twins[0].set(ylabel=r"$\Re(\mathrm{envelope})$")
twins[1].set(ylabel=r"$\Im(\mathrm{envelope})$")
fig.tight_layout()
plt.show()
