import numpy as np
import pytest

from finite_dipole.multilayer import (
    _beta_and_t_stack_from_inputs,
    _beta_func_from_stack,
)


def test_beta_and_t_stack_from_inputs_error_when_eps_and_beta_are_None():
    with pytest.raises(
        ValueError, match="Either `eps_stack` or `beta_stack` must be specified."
    ):
        _beta_and_t_stack_from_inputs(eps_stack=None, beta_stack=None, t_stack=None)


VALID_EPS_AND_T_STACK_PAIRS = [
    ([1, 2], []),
    ([1, 2], None),
    ([1, 2, 3], [1]),
    ([1, 2, 3, 4], [1, 2]),
    ([[1], [2]], []),
    ([[1], [2]], None),
    ([[1, 2], [3, 4]], None),
    ([[1, 2], [3, 4]], []),
    ([[1, 2], [3, 4]], None),
]


@pytest.mark.parametrize("eps_stack, t_stack", VALID_EPS_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_eps_creates_right_shape(eps_stack, t_stack):
    beta_stack, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=eps_stack, beta_stack=None, t_stack=t_stack
    )
    assert (
        np.shape(eps_stack)[0]
        == np.shape(beta_stack)[0] + 1
        == np.shape(t_stack_new)[0] + 2
    )


@pytest.mark.parametrize("eps_stack, t_stack", VALID_EPS_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_eps_returns_arrays(eps_stack, t_stack):
    beta_stack, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=eps_stack, beta_stack=None, t_stack=t_stack
    )
    assert type(beta_stack) == type(t_stack_new) == np.ndarray


@pytest.mark.parametrize("eps_stack, t_stack", VALID_EPS_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_eps_leaves_t_unchanged(eps_stack, t_stack):
    beta_stack, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=eps_stack, beta_stack=None, t_stack=t_stack
    )
    np.testing.assert_equal(t_stack_new, t_stack)


VALID_BETA_AND_T_STACK_PAIRS = [
    ([1], []),
    ([1], None),
    ([1, 2], [1]),
    ([1, 2, 3], [1, 2]),
    ([[1]], []),
    ([[1]], None),
    ([[1, 2]], []),
    ([[1, 2]], None),
    ([[1, 2], [3, 4]], [1]),
]


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_creates_right_shape(beta_stack, t_stack):
    beta_stack_new, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    assert np.shape(beta_stack_new)[0] == np.shape(t_stack_new)[0] + 1


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_returns_arrays(beta_stack, t_stack):
    beta_stack_new, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    assert type(beta_stack_new) == type(t_stack_new) == np.ndarray


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_leaves_t_unchanged(beta_stack, t_stack):
    beta_stack_new, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    np.testing.assert_equal(t_stack_new, t_stack)


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_leaves_beta_unchanged(beta_stack, t_stack):
    beta_stack_new, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    np.testing.assert_equal(beta_stack_new, beta_stack)


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_func_from_stack_creates_right_shape(beta_stack, t_stack):
    beta_k = _beta_func_from_stack(np.asarray(beta_stack), np.asarray(t_stack))
    assert np.shape(beta_k) == np.shape(beta_stack)[1:]
