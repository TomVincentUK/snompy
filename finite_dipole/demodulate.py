"""
Demodulation code.
"""
import numpy as np
from scipy.integrate import trapezoid, simpson
from numba import njit
from numba.extending import is_jitted


def _sampled_integrand(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    x = x_0 + x_amplitude * np.cos(theta)
    f = f_x(x, *f_args)
    envelope = np.exp(-1j * harmonic * theta)
    return f * envelope


_sampled_integrand_compiled = njit(_sampled_integrand)


def demod(
    f_x,
    x_0,
    x_amplitude,
    harmonic,
    f_args=(),
    method="trapezium",
    n_samples=65,
):
    if method not in ["trapezium", "simpson", "adaptive"]:
        raise ValueError("`method` must be 'trapezium', 'simpson' or 'adaptive'.")

    if method == "adaptive":
        raise NotImplementedError()
    else:
        x_0, x_amplitude, harmonic, *f_args = [
            np.array(arr)[..., np.newaxis]
            for arr in np.broadcast_arrays(*(x_0, x_amplitude, harmonic) + f_args)
        ]
        print(x_0.shape)

        # Function falls back to uncompiled code if not given jitted f_x
        if is_jitted(f_x):
            si = _sampled_integrand_compiled
            f_args = tuple(f_args)
        else:
            si = _sampled_integrand
        integrand = si(f_x, x_0, x_amplitude, harmonic, f_args, n_samples)

        if method == "trapezium":
            result = trapezoid(integrand) / (n_samples - 1)
        elif method == "simpson":
            result = simpson(integrand) / (n_samples - 1)

    return result
