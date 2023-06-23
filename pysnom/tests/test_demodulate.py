import numpy as np

from pysnom.demodulate import demod


def test_only_zeroth_harmonic_with_constant_function():
    def constant_function(x):
        return np.ones_like(x)

    result = demod(
        f_x=constant_function,
        x_0=0,
        x_amplitude=1,
        n=np.arange(3),
    )
    np.testing.assert_almost_equal(result, np.array([1, 0, 0]))


def test_only_first_harmonic_with_linear_function():
    def linear_function(x):
        return 2 * x

    result = demod(f_x=linear_function, x_0=0, x_amplitude=1, n=np.arange(3))
    np.testing.assert_almost_equal(result, [0, 1, 0])


def test_broadcasting():
    def function_with_args(x, a, b, c):
        return a * x**2 + b * x + c

    x_0 = 0
    x_amplitude = np.arange(2)
    n = np.arange(3)[:, np.newaxis]
    a = np.arange(4)[:, np.newaxis, np.newaxis]
    b = np.arange(5)[:, np.newaxis, np.newaxis, np.newaxis]
    c = np.arange(6)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    target_shape = (x_0 + x_amplitude + n + a + b + c).shape

    result = demod(
        f_x=function_with_args,
        x_0=x_0,
        x_amplitude=x_amplitude,
        n=n,
        f_args=(a, b, c),
    )

    assert result.shape == target_shape
