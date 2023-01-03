import numpy as np
import pytest
from numba import njit

from finite_dipole.demodulate import demod

METHODS = "trapezium", "simpson", "adaptive"


@pytest.mark.parametrize("method", METHODS)
def test_only_zeroth_harmonic_with_constant_function(method):
    def constant_function(x):
        return np.ones_like(x)

    result = demod(
        f_x=constant_function,
        x_0=0,
        x_amplitude=1,
        harmonic=np.arange(3),
        method=method,
    )
    np.testing.assert_almost_equal(result, np.array([1, 0, 0]))


@pytest.mark.parametrize("method", METHODS)
def test_only_first_harmonic_with_linear_function(method):
    def linear_function(x):
        return 2 * x

    result = demod(
        f_x=linear_function, x_0=0, x_amplitude=1, harmonic=np.arange(3), method=method
    )
    np.testing.assert_almost_equal(result, [0, 1, 0])


@pytest.mark.parametrize("method", METHODS)
def test_jitted_and_nonjitted_f_x(method):
    def nonjitted_function(x):
        return np.exp(-x)

    jitted_function = njit(nonjitted_function)

    nonjitted_result = demod(
        f_x=nonjitted_function, x_0=0, x_amplitude=1, harmonic=1, method=method
    )
    jitted_result = demod(
        f_x=jitted_function, x_0=0, x_amplitude=1, harmonic=1, method=method
    )
    assert nonjitted_result == jitted_result


@pytest.mark.parametrize("method", METHODS)
def test_broadcasting(method):
    def function_with_args(x, a, b, c):
        return a * x**2 + b * x + c

    x_0 = 0
    x_amplitude = np.arange(2)
    harmonic = np.arange(3)[:, np.newaxis]
    a = np.arange(4)[:, np.newaxis, np.newaxis]
    b = np.arange(5)[:, np.newaxis, np.newaxis, np.newaxis]
    c = np.arange(6)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    target_shape = (x_0 + x_amplitude + harmonic + a + b + c).shape

    result = demod(
        f_x=function_with_args,
        x_0=x_0,
        x_amplitude=x_amplitude,
        harmonic=harmonic,
        method=method,
        f_args=(a, b, c),
    )

    assert result.shape == target_shape


@pytest.mark.parametrize("method", METHODS)
def test_jitted_broadcasting(method):
    @njit
    def jitted_function_with_args(x, a, b, c):
        return a * x**2 + b * x + c

    x_0 = 0
    x_amplitude = np.arange(2)
    harmonic = np.arange(3)[:, np.newaxis]
    a = np.arange(4)[:, np.newaxis, np.newaxis]
    b = np.arange(5)[:, np.newaxis, np.newaxis, np.newaxis]
    c = np.arange(6)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    target_shape = (x_0 + x_amplitude + harmonic + a + b + c).shape

    result = demod(
        f_x=jitted_function_with_args,
        x_0=x_0,
        x_amplitude=x_amplitude,
        harmonic=harmonic,
        method=method,
        f_args=(a, b, c),
    )

    assert result.shape == target_shape
