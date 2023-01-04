import numpy as np
import pytest

from finite_dipole.bulk import eff_pol
from finite_dipole.multilayer import (
    _beta_and_t_stack_from_inputs,
    _beta_func_from_stack,
    eff_pol_ML,
)

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


def test_beta_and_t_stack_from_inputs_error_when_eps_and_beta_are_None():
    with pytest.raises(
        ValueError, match="Either `eps_stack` or `beta_stack` must be specified."
    ):
        _beta_and_t_stack_from_inputs(eps_stack=None, beta_stack=None, t_stack=None)


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
def test_beta_and_t_stack_from_inputs_beta_leaves_beta_unchanged(beta_stack, t_stack):
    beta_stack_new, t_stack_new = _beta_and_t_stack_from_inputs(
        eps_stack=None, beta_stack=beta_stack, t_stack=t_stack
    )
    np.testing.assert_equal(beta_stack_new, beta_stack)


@pytest.mark.parametrize("beta_stack, t_stack", VALID_BETA_AND_T_STACK_PAIRS)
def test_beta_func_from_stack_creates_right_shape(beta_stack, t_stack):
    beta_k = _beta_func_from_stack(np.asarray(beta_stack), np.asarray(t_stack))
    assert np.shape(beta_k(1)) == np.shape(beta_stack)[1:]


def test_eff_pol_ML_broadcasting():
    # Measurement parameters
    wavenumber = np.linspace(1680, 1780, 16) * 1e2
    z = 50e-9
    tapping_amplitude = 50e-9
    harmonic = np.arange(2, 5)[:, np.newaxis]
    thickness = np.arange(1, 100, 8)[:, np.newaxis, np.newaxis] * 1e-9

    # Eventual output shape should match broadcast arrays
    target_shape = (wavenumber + z + tapping_amplitude + harmonic + thickness).shape

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

    alpha_eff = eff_pol_ML(
        z=z,
        tapping_amplitude=tapping_amplitude,
        harmonic=harmonic,
        eps_stack=(eps_super, eps_middle, eps_sub),
        t_stack=(thickness,),
    )
    assert alpha_eff.shape == target_shape


def test_eff_pol_ML_approach_curve_decays():
    alpha_eff = eff_pol_ML(
        z=np.linspace(0, 100, 16) * 1e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5)[:, np.newaxis],
        eps_stack=(1, 2 + 1j, 11.7),
        t_stack=(100e-9,),
    )
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


def test_eff_pol_ML_harmonics_decay():
    alpha_eff = eff_pol_ML(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5),
        eps_stack=(1, 2 + 1j, 11.7),
        t_stack=(100e-9,),
    )
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


def test_eff_pol_ML_zero_thickness_layer_invisible():
    alpha_eff = eff_pol_ML(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5),
        eps_stack=(1, 2 + 1j, 11.7),
        t_stack=(100e-9,),
    )
    alpha_eff_with_zero_thickness_layer = eff_pol_ML(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5),
        eps_stack=(1, 2 + 1j, 3, 11.7),
        t_stack=(100e-9, 0),
    )
    np.testing.assert_almost_equal(alpha_eff, alpha_eff_with_zero_thickness_layer)


def test_eff_pol_ML_two_layers_same_as_bulk():
    alpha_eff_bulk = eff_pol(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5),
        eps_sample=11.7,
    )
    alpha_eff_ML = eff_pol_ML(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 5),
        eps_stack=(1, 11.7),
    )
    np.testing.assert_almost_equal(alpha_eff_bulk, alpha_eff_ML)
