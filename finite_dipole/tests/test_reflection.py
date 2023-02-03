import numpy as np
import pytest

from finite_dipole.reflection import interface_stack, refl_coeff_ML

VALID_EPS_AND_T_STACK_PAIRS = [
    ([1, 2], []),
    ([1, 2], None),
    ([1, 2, 3], [1]),
    ([1, 2, 3, 4], [1, 2]),
    ([[1], [2]], []),
    ([[1, 2], [3, 4]], []),
]

VALID_BETA_AND_T_STACK_PAIRS = [
    ([1], []),
    ([1], None),
    ([1, 2], [1]),
    ([1, 2, 3], [1, 2]),
    ([[1]], []),
    ([[1, 2]], []),
    ([[1, 2], [3, 4]], [1]),
]


def test_beta_and_t_stack_from_inputs_error_when_eps_and_beta_are_None():
    with pytest.raises(
        ValueError, match="Either `eps_stack` or `beta_stack` must be specified."
    ):
        interface_stack(eps_stack=None, beta_stack=None, t_stack=None)


@pytest.mark.parametrize("eps_stack, t_stack", VALID_EPS_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_eps_creates_right_shape(eps_stack, t_stack):
    beta_stack, t_stack_new = interface_stack(
        eps_stack=eps_stack, beta_stack=None, t_stack=t_stack
    )
    assert (
        np.shape(eps_stack)[0]
        == np.shape(beta_stack)[0] + 1
        == np.shape(t_stack_new)[0] + 2
    )


@pytest.mark.parametrize("eps_stack, t_stack", VALID_EPS_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_eps_returns_arrays(eps_stack, t_stack):
    beta_stack, t_stack_new = interface_stack(
        eps_stack=eps_stack, beta_stack=None, t_stack=t_stack
    )
    assert type(beta_stack) == type(t_stack_new) == np.ndarray


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_creates_right_shape(beta_stack, t_stack):
    beta_stack_new, t_stack_new = interface_stack(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    assert np.shape(beta_stack_new)[0] == np.shape(t_stack_new)[0] + 1


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_returns_arrays(beta_stack, t_stack):
    beta_stack_new, t_stack_new = interface_stack(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    assert type(beta_stack_new) == type(t_stack_new) == np.ndarray


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_and_t_stack_from_inputs_beta_leaves_beta_unchanged(beta_stack, t_stack):
    beta_stack_new, t_stack_new = interface_stack(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    np.testing.assert_equal(beta_stack_new, beta_stack)


def test_refl_coeff_ML_broadcasting():
    # Measurement parameters
    k = np.linspace(0, 1, 128)
    wavenumber = np.linspace(1680, 1780, 16)[:, np.newaxis] * 1e2
    layer_thickness = np.arange(1, 100, 8)[:, np.newaxis, np.newaxis] * 1e-9

    # Constant sub- and superstrate dielectric functions
    eps_super = 1
    eps_sub = 11.7

    # Dispersive middle layer dielectric function
    eps_inf = 2
    osc_freq = 1740e2
    osc_width = 20e2
    osc_strength = 15e-3
    eps_middle = eps_inf + (osc_strength * osc_freq**2) / (
        osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
    )

    beta_stack, t_stack = interface_stack(
        eps_stack=(eps_super, eps_middle, eps_sub), t_stack=(layer_thickness,)
    )
    target_shape = (k + beta_stack[0] + t_stack[0]).shape
    assert refl_coeff_ML(k, beta_stack, t_stack).shape == target_shape


def test_interface_stack_warns_for_zero_thickness():
    with pytest.warns(
        UserWarning,
        match=" ".join(
            [
                "`t_stack` contains zeros.",
                "Zero-thickness dielectric layers are unphysical.",
                "Results may not be as expected.",
            ]
        ),
    ):
        interface_stack(beta_stack=(0.5, 0.5), t_stack=(0,))


def test_interface_stack_warning_if_eps_and_beta():
    with pytest.warns(
        UserWarning, match="`beta_stack` overrides `eps_stack` when both are specified."
    ):
        interface_stack(eps_stack=(1, 2, 3), beta_stack=(0.5, 0.5), t_stack=(1,))


def test_interface_stack_error_if_no_material():
    with pytest.raises(Exception) as e:
        interface_stack()
    assert e.type == ValueError
    assert "Either `eps_stack` or `beta_stack` must be specified." in str(e.value)


def test_interface_stack_error_if_mismatched_eps_t():
    with pytest.raises(Exception) as e:
        interface_stack(eps_stack=(1, 2, 3, 4), t_stack=(1,))
    assert e.type == ValueError
    assert "`eps_stack` must be 2 longer than `t_stack` along the first axis." in str(
        e.value
    )


def test_interface_stack_error_if_mismatched_beta_t():
    with pytest.raises(Exception) as e:
        interface_stack(beta_stack=(0.5, 0.5, 0.5), t_stack=(1,))
    assert e.type == ValueError
    assert "`beta_stack` must be 1 longer than `t_stack` along the first axis." in str(
        e.value
    )
