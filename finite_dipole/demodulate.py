"""
Demodulation code.
"""
import numpy as np
from scipy.integrate import trapezoid, simpson, quad_vec
from numba import njit
from numba.extending import is_jitted


def _sampled_integrand(f_x, x_0, x_amplitude, harmonic, f_args, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    x = x_0 + x_amplitude * np.cos(theta)
    f = f_x(x, *f_args)
    envelope = np.exp(-1j * harmonic * theta)
    return f * envelope


_sampled_integrand_compiled = njit(_sampled_integrand)


def _generate_f_theta(f_x, x_0, x_amplitude, harmonic, f_args):
    def f_theta(theta, x_0, x_amplitude, harmonic, *f_args):
        x = x_0 + x_amplitude * np.cos(theta)
        f = f_x(x, *f_args)
        envelope = np.exp(-1j * harmonic * theta)
        return f * envelope

    return njit(f_theta) if is_jitted(f_x) else f_theta


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
        x_0, x_amplitude, harmonic, *f_args = [
            np.array(arr)
            for arr in np.broadcast_arrays(*(x_0, x_amplitude, harmonic) + f_args)
        ]
        f_args = tuple(f_args)

        f_theta = _generate_f_theta(f_x, x_0, x_amplitude, harmonic, f_args)

        result, _ = quad_vec(
            lambda t: f_theta(t, x_0, x_amplitude, harmonic, *f_args), -np.pi, np.pi
        )
        result /= 2 * np.pi
    else:
        x_0, x_amplitude, harmonic, *f_args = [
            np.array(arr)[
                ..., np.newaxis
            ]  # extra dimension added to arrays for _sampled_integrand to extend along
            for arr in np.broadcast_arrays(*(x_0, x_amplitude, harmonic) + f_args)
        ]
        f_args = tuple(f_args)

        # Function falls back to uncompiled code if not given jitted f_x
        si = _sampled_integrand_compiled if is_jitted(f_x) else _sampled_integrand
        integrand = si(f_x, x_0, x_amplitude, harmonic, f_args, n_samples)

        if method == "trapezium":
            result = trapezoid(integrand) / (n_samples - 1)
        elif method == "simpson":
            result = simpson(integrand) / (n_samples - 1)

    return result
