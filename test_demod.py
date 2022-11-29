"""
Test of finite dipole model (FDM) for finding a dielectric function of a
fictitious material from a simulated SNOM measurement.

References
----------
.. [1] Z. M. Zhang, G. Lefever-Button, F. R. Powell,
   Infrared refractive index and extinction coefficient of polyimide films,
   Int. J. Thermophys., 19 (1998) 905.
   https://doi.org/10.1023/A:1022655309574.
.. [2] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W.
   Alexander, C.A. Ward,
   Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
   Ti, and W in the infrared and far infrared,
   Appl. Opt. 22 (1983) 1099.
   https://doi.org/10.1364/AO.22.001099.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

from finite_dipole import _eff_pol_new as eff_pol


def eps_Lorentz(omega, eps_inf, omega_0, strength, gamma):
    """
    Lorentzian oscillator dielectric function model. Function definition
    from equation (5) of reference [1]_.
    """
    return eps_inf + (strength * omega_0**2) / (
        omega_0**2 - omega**2 - 1j * gamma * omega
    )


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (2)
    of reference [2]_.
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = np.linspace(1000, 1250, 129) * 1e2
z_0 = 50e-9
tapping_amplitude = 50e-9
radius = 20e-9
harmonic = 3
amp_const = 100  # Represents the overall signal level of the SNOM measurement
noise_level = 1e-3  # Sigma of Gaussian noise added to both real and imaginary parts

# fictitious material
eps_X = eps_Lorentz(wavenumber, 10, 1100e2, 1, 10e2)
alpha_X_n = eff_pol(
    z_0,
    tapping_amplitude,
    harmonic,
    eps_sample=eps_X,
    radius=radius,
    demod_method="trapezium",
)
SNOM_X = amp_const * alpha_X_n + noise_level * (
    np.random.randn(*alpha_X_n.shape) + 1j * np.random.randn(*alpha_X_n.shape)
)

# Using gold as a reference material
eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]_
alpha_Au_n = eff_pol(
    z_0,
    tapping_amplitude,
    harmonic,
    eps_sample=eps_Au,
    radius=radius,
    demod_method="trapezium",
)
SNOM_Au = amp_const * alpha_Au_n + noise_level * (
    np.random.randn(*alpha_Au_n.shape) + 1j * np.random.randn(*alpha_Au_n.shape)
)

SNOM_ratio = SNOM_X / SNOM_Au


def min_full(L_params):
    """Minimization function for Lorentzian dielectric function fit"""
    ratio = (
        eff_pol(
            z_0,
            tapping_amplitude,
            harmonic,
            eps_sample=eps_Lorentz(wavenumber, *L_params),
            radius=radius,
            demod_method="trapezium",
        )
        / alpha_Au_n
    )
    return np.abs(SNOM_ratio - ratio).sum()


res = minimize(fun=min_full, x0=(10, 1100e2, 1, 10e2))
L_params = res.x
start_est = eps_Lorentz(wavenumber, *L_params)
alpha_fit = (
    eff_pol(
        z_0,
        tapping_amplitude,
        harmonic,
        eps_sample=start_est,
        radius=radius,
        demod_method="trapezium",
    )
    * amp_const
)


def min_pointwise(eps, SNOM_ratio, eps_Au):
    """Minimization function for point-wise dielectric function fit"""
    eps = eps[0] + 1j * eps[1]
    alpha_X_n = eff_pol(
        z_0,
        tapping_amplitude,
        harmonic,
        eps_sample=eps,
        radius=radius,
        demod_method="trapezium",
    )
    alpha_Au_n = eff_pol(
        z_0,
        tapping_amplitude,
        harmonic,
        eps_sample=eps_Au,
        radius=radius,
        demod_method="trapezium",
    )
    return np.abs(SNOM_ratio - alpha_X_n / alpha_Au_n)


fit = np.empty_like(eps_Au)
for i, (_ratio, _eps_Au, _start) in tqdm(
    enumerate(zip(SNOM_ratio, eps_Au, start_est)), total=eps_Au.size
):
    res = minimize(
        fun=min_pointwise,
        x0=(_start.real, _start.imag),
        args=(_ratio, _eps_Au),
        method="Nelder-Mead",
        tol=1e-12,
    )
    fit[i] = res.x[0] + 1j * res.x[1]

fig, axes = plt.subplots(nrows=2, sharex=True)
c_re = "C0"
c_im = "C1"
c_amp = "C2"
c_phase = "C3"
line_alpha = 0.75

re_ax = axes[0]
re_ax.scatter(wavenumber * 1e-2, np.real(eps_X), c=c_re, label="true value")
re_ax.scatter(wavenumber * 1e-2, np.imag(eps_X), c=c_im)
re_ax.plot(
    wavenumber * 1e-2,
    np.real(start_est),
    c=c_re,
    ls="--",
    alpha=line_alpha,
    label="Lorentzian fit",
)
re_ax.plot(wavenumber * 1e-2, np.imag(start_est), c=c_im, ls="--", alpha=line_alpha)
re_ax.plot(
    wavenumber * 1e-2,
    np.real(fit),
    c=c_re,
    ls="-",
    alpha=line_alpha,
    label="pointwise fit",
)
re_ax.plot(wavenumber * 1e-2, np.imag(fit), c=c_im, ls="-", alpha=line_alpha)
re_ax.axhline(0, lw=plt.rcParams["axes.linewidth"], c="k")


amp_ax = axes[1]
phase_ax = amp_ax.twinx()
amp_ax.scatter(wavenumber * 1e-2, np.abs(SNOM_X), c=c_amp, label="simulated SNOM")
phase_ax.scatter(wavenumber * 1e-2, np.unwrap(np.angle(SNOM_X)), c=c_phase)
amp_ax.plot(
    wavenumber * 1e-2,
    np.abs(alpha_fit),
    c=c_amp,
    ls="--",
    alpha=line_alpha,
    label="Lorentzian fit",
)
phase_ax.plot(
    wavenumber * 1e-2,
    np.unwrap(np.angle(alpha_fit)),
    c=c_phase,
    ls="--",
    alpha=line_alpha,
)
amp_ax.set(
    xlabel=r"$\omega$ / cm$^{-1}$",
    xlim=(wavenumber.max() * 1e-2, wavenumber.min() * 1e-2),
    ylim=(0, None),
)

im_ax = re_ax.twinx()  # Only for the sake of labelling
im_ax.set_ylim(re_ax.get_ylim())
re_ax.set_ylabel(r"$\mathrm{Re}\left(\varepsilon_{sample}\right)$", color=c_re)
im_ax.set_ylabel(r"$\mathrm{Im}\left(\varepsilon_{sample}\right)$", color=c_im)
amp_ax.set_ylabel(r"$s_{" f"{harmonic}" r"}$ / a.u.", color=c_amp)
phase_ax.set_ylabel(r"$\phi_{" f"{harmonic}" r"}$", color=c_phase)
re_ax.legend()
amp_ax.legend()

fig.tight_layout()
plt.show(block=False)
