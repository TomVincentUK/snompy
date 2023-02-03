import numpy as np
from scipy.integrate import quad_vec

from finite_dipole.bulk import eff_pol
from finite_dipole.multilayer import (
    eff_pol_0_ML,
    eff_pol_ML,
    eff_pos_and_charge,
    phi_E_0,
)
from finite_dipole.reflection import interface_stack, refl_coeff_ML

# Measurement parameters
Z = 40e-9
Z_Q = 60e-9
Z_APPROACH = np.linspace(0, 100, 101) * 1e-9

# Beta and t for single calculation
BETA_STACK_SINGLE, T_STACK_SINGLE = interface_stack(
    eps_stack=(1, 2 + 1j, 11.7), t_stack=(100e-9,)
)


# Dispersive beta and different t
eps_super = 1
eps_sub = 11.7
eps_inf = 2
osc_freq = 1740e2
osc_width = 20e2
osc_strength = 15e-3
wavenumber = np.linspace(1680, 1780, 256) * 1e2
eps_middle = eps_inf + (osc_strength * osc_freq**2) / (
    osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
)
thickness = np.linspace(2, 100, 50)[:, np.newaxis] * 1e-9
BETA_STACK_VECTOR, T_STACK_VECTOR = interface_stack(
    eps_stack=(eps_super, eps_middle, eps_sub), t_stack=(thickness,)
)

# Demodulation parameters
TAPPING_AMPLITUDE = 20e-9
HARMONIC = np.arange(2, 5)[:, np.newaxis, np.newaxis]


def test_phi_E_0_integrals():
    phi, E = phi_E_0(Z_Q, BETA_STACK_SINGLE, T_STACK_SINGLE)

    phi_scipy, _ = quad_vec(
        lambda x: refl_coeff_ML(x / (2 * Z_Q), BETA_STACK_SINGLE, T_STACK_SINGLE)
        * np.exp(-x),
        0,
        np.inf,
    )
    phi_scipy /= 2 * Z_Q
    np.testing.assert_allclose(phi, phi_scipy)

    E_scipy, _ = quad_vec(
        lambda x: refl_coeff_ML(x / (2 * Z_Q), BETA_STACK_SINGLE, T_STACK_SINGLE)
        * x
        * np.exp(-x),
        0,
        np.inf,
    )
    E_scipy /= 4 * Z_Q**2
    np.testing.assert_allclose(E, E_scipy)


def test_eff_pos_and_charge_broadcasting():
    target_shape = (Z_Q * BETA_STACK_VECTOR[0] * T_STACK_VECTOR[0]).shape
    z_image, beta_image = eff_pos_and_charge(Z_Q, BETA_STACK_VECTOR, T_STACK_VECTOR)
    assert z_image.shape == beta_image.shape == target_shape


def test_eff_pol_0_ML_broadcasting():
    target_shape = (Z * BETA_STACK_VECTOR[0] * T_STACK_VECTOR[0]).shape
    alpha_eff = eff_pol_0_ML(Z, BETA_STACK_VECTOR, T_STACK_VECTOR)
    assert alpha_eff.shape == target_shape


def test_eff_pol_ML_broadcasting():
    target_shape = (
        Z * BETA_STACK_VECTOR[0] * T_STACK_VECTOR[0] * TAPPING_AMPLITUDE * HARMONIC
    ).shape
    alpha_eff = eff_pol_ML(
        z=Z,
        tapping_amplitude=TAPPING_AMPLITUDE,
        harmonic=HARMONIC,
        beta_stack=BETA_STACK_VECTOR,
        t_stack=T_STACK_VECTOR,
    )
    assert alpha_eff.shape == target_shape


def test_eff_pol_0_ML_approach_curve_decays():
    alpha_eff = eff_pol_0_ML(Z_APPROACH, BETA_STACK_SINGLE, T_STACK_SINGLE)
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


def test_eff_pol_ML_approach_curve_decays():
    alpha_eff = eff_pol_ML(
        z=Z_APPROACH,
        tapping_amplitude=TAPPING_AMPLITUDE,
        harmonic=HARMONIC,
        beta_stack=BETA_STACK_SINGLE,
        t_stack=T_STACK_SINGLE,
    )
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


def test_eff_pol_ML_harmonics_decay():
    alpha_eff = eff_pol_ML(
        z=Z,
        tapping_amplitude=TAPPING_AMPLITUDE,
        harmonic=HARMONIC,
        beta_stack=BETA_STACK_SINGLE,
        t_stack=T_STACK_SINGLE,
    )
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


def test_eff_pol_ML_two_layers_same_as_bulk():
    eps_stack = 1, 11.7
    alpha_eff_bulk = eff_pol(
        z=Z,
        tapping_amplitude=TAPPING_AMPLITUDE,
        harmonic=HARMONIC,
        eps_sample=eps_stack[-1],
        eps_environment=eps_stack[0],
    )
    alpha_eff_ML = eff_pol_ML(
        z=Z,
        tapping_amplitude=TAPPING_AMPLITUDE,
        harmonic=HARMONIC,
        eps_stack=eps_stack,
    )
    np.testing.assert_almost_equal(alpha_eff_bulk, alpha_eff_ML)
