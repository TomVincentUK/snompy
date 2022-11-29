"""
Demodulation code.
"""
import numpy as np
from scipy.integrate import trapezoid, simpson
from numba import njit


@njit
def _sampled_integrand(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    x = x_0 + x_amplitude * np.cos(theta)
    f = f_x(x, *f_args)
    envelope = np.exp(-1j * harmonic * theta)
    return f * envelope


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

    x_0, x_amplitude, harmonic, *f_args = np.broadcast_arrays(
        *(x_0, x_amplitude, harmonic) + f_args
    )
    f_args = tuple(f_args)

    if method == "adaptive":
        raise NotImplementedError()
    else:
        x_0, x_amplitude, harmonic, *f_args = [
            np.array(arr)[..., np.newaxis]
            for arr in np.broadcast_arrays(*(x_0, x_amplitude, harmonic) + f_args)
        ]
        f_args = tuple(f_args)

        integrand = _sampled_integrand(
            f_x, x_0, x_amplitude, harmonic, f_args, n_samples
        )

        if method == "trapezium":
            result = trapezoid(integrand) / (n_samples - 1)
        elif method == "simpson":
            result = simpson(integrand) / (n_samples - 1)

    return result
