import numpy as np
from scipy.integrate import quad_vec

from finite_dipole.bulk import eff_pol
from finite_dipole.multilayer import (
    eff_pol_0_ML,
    eff_pol_ML,
    eff_pos_and_charge,
    phi_E_0,
)
from finite_dipole.reflection import beta_and_t_stack_from_inputs, refl_coeff_ML


def test_phi_E_0_integrals():
    z_q = 20e-9
    beta_stack, t_stack = beta_and_t_stack_from_inputs(
        eps_stack=(1, 2 + 1j, 11.7), t_stack=(100e-9,)
    )
    phi, E = phi_E_0(z_q, beta_stack, t_stack)

    phi_scipy, _ = quad_vec(
        lambda x: refl_coeff_ML(x / (2 * z_q), beta_stack, t_stack) * np.exp(-x),
        0,
        np.inf,
    )
    phi_scipy /= 2 * z_q
    np.testing.assert_allclose(phi, phi_scipy)

    E_scipy, _ = quad_vec(
        lambda x: refl_coeff_ML(x / (2 * z_q), beta_stack, t_stack) * x * np.exp(-x),
        0,
        np.inf,
    )
    E_scipy /= 4 * z_q**2
    np.testing.assert_allclose(E, E_scipy)


def test_eff_charge_and_pos_broadcasting():
    # Measurement parameters
    z_q = 20e-9
    wavenumber = np.linspace(1680, 1780, 16) * 1e2
    thickness = np.arange(1, 100, 8)[:, np.newaxis] * 1e-9

    # Eventual output shape should match broadcast arrays
    target_shape = (z_q + wavenumber + thickness).shape

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

    beta_stack, t_stack = beta_and_t_stack_from_inputs(
        eps_stack=(eps_super, eps_middle, eps_sub), t_stack=(thickness,)
    )
    z_image, beta_image = eff_pos_and_charge(
        z_q, beta_stack, t_stack, *np.polynomial.laguerre.laggauss(64)
    )
    assert z_image.shape == beta_image.shape == target_shape


def test_eff_pol_0_ML_broadcasting():
    # Measurement parameters
    wavenumber = np.linspace(1680, 1780, 16) * 1e2
    z = 50e-9
    thickness = np.arange(1, 100, 8)[:, np.newaxis] * 1e-9

    # Eventual output shape should match broadcast arrays
    target_shape = (wavenumber + z + thickness).shape

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

    beta_stack, t_stack = beta_and_t_stack_from_inputs(
        eps_stack=(eps_super, eps_middle, eps_sub), t_stack=(thickness,)
    )
    alpha_eff = eff_pol_0_ML(z, beta_stack, t_stack)
    assert alpha_eff.shape == target_shape


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


def test_eff_pol_0_ML_approach_curve_decays():
    z = np.linspace(0, 100, 16) * 1e-9
    beta_stack, t_stack = beta_and_t_stack_from_inputs(
        eps_stack=(1, 2 + 1j, 11.7), t_stack=(100e-9,)
    )
    alpha_eff = eff_pol_0_ML(z, beta_stack, t_stack)
    assert (np.diff(np.abs(alpha_eff)) < 0).all()


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
