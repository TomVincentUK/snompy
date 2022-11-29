import pytest
import numpy as np
from numba import njit
from finite_dipole.demodulate import demod

methods = "trapezium", "simpson"


@pytest.mark.parametrize("method", methods)
def test_only_zeroth_harmonic_with_constant_function(method):
    @njit
    def constant_function(x):
        return np.ones_like(x)

    result = demod(
        f_x=constant_function,
        x_0=0,
        x_amplitude=1,
        harmonic=np.arange(3),
        method=method,
    )
    np.testing.assert_almost_equal(result, [1, 0, 0])


@pytest.mark.parametrize("method", methods)
def test_only_first_harmonic_with_linear_function(method):
    @njit
    def linear_function(x):
        return x

    result = demod(
        f_x=linear_function, x_0=0, x_amplitude=1, harmonic=np.arange(3), method=method
    )
    np.testing.assert_almost_equal(result, [0, 0.5, 0])
