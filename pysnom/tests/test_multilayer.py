import numpy as np
from scipy.integrate import quad_vec

from pysnom.fdm import bulk, multi
from pysnom.sample import Sample

# Measurement parameters
Z = 40e-9
Z_Q = 60e-9
Z_APPROACH = np.linspace(0, 100, 101) * 1e-9

# Beta and t for single calculation
SAMPLE_SINGLE = Sample(eps_stack=(1, 2 + 1j, 11.7), t_stack=(100e-9,))

# Dispersive beta and different t
eps_env = 1
eps_substrate = 11.7
eps_inf = 2
osc_freq = 1738e2
osc_width = 20e2
osc_strength = 14e-3
wavenumber = np.linspace(1680, 1780, 128) * 1e2
eps_middle = eps_inf + (osc_strength * osc_freq**2) / (
    osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
)  # Lorentzian oscillator
thickness = np.geomspace(10, 100, 32)[:, np.newaxis] * 1e-9
SAMPLE_VECTOR = Sample(
    eps_stack=(eps_env, eps_middle, eps_substrate), t_stack=(thickness,)
)

# Demodulation parameters
TAPPING_AMPLITUDE = 20e-9
HARMONIC = np.arange(2, 5)[:, np.newaxis, np.newaxis]


def test_phi_E_0_integrals():
    phi, E = multi.phi_E_0(Z_Q, SAMPLE_SINGLE)

    phi_scipy, _ = quad_vec(
        lambda x: SAMPLE_SINGLE.refl_coef_qs(x / (2 * Z_Q)) * np.exp(-x),
        0,
        np.inf,
    )
    phi_scipy /= 2 * Z_Q
    np.testing.assert_allclose(phi, phi_scipy)

    E_scipy, _ = quad_vec(
        lambda x: SAMPLE_SINGLE.refl_coef_qs(x / (2 * Z_Q)) * x * np.exp(-x),
        0,
        np.inf,
    )
    E_scipy /= 4 * Z_Q**2
    np.testing.assert_allclose(E, E_scipy)


def test_eff_pos_and_charge_broadcasting():
    target_shape = (Z_Q * SAMPLE_VECTOR.eps_stack[0] * SAMPLE_VECTOR.t_stack[0]).shape
    z_image, beta_image = multi.eff_pos_and_charge(Z_Q, SAMPLE_VECTOR)
    assert z_image.shape == beta_image.shape == target_shape


def test_eff_pol_multi_broadcasting():
    target_shape = (Z * SAMPLE_VECTOR.eps_stack[0] * SAMPLE_VECTOR.t_stack[0]).shape
    alpha_eff = multi.eff_pol(Z, SAMPLE_VECTOR)
    assert alpha_eff.shape == target_shape


def test_eff_pol_n_multi_broadcasting():
    target_shape = (
        Z
        * SAMPLE_VECTOR.eps_stack[0]
        * SAMPLE_VECTOR.t_stack[0]
        * TAPPING_AMPLITUDE
        * HARMONIC
    ).shape
    alpha_eff = multi.eff_pol_n(
        z_tip=Z,
        A_tip=TAPPING_AMPLITUDE,
        n=HARMONIC,
        sample=SAMPLE_VECTOR,
    )
    assert alpha_eff.shape == target_shape


def test_eff_pol_n_multi_two_layers_same_as_bulk():
    sample = Sample(eps_stack=(1, 10))
    alpha_eff_bulk = bulk.eff_pol_n(
        z_tip=Z,
        A_tip=TAPPING_AMPLITUDE,
        n=HARMONIC,
        sample=sample,
    )
    alpha_eff_multi = multi.eff_pol_n(
        z_tip=Z,
        A_tip=TAPPING_AMPLITUDE,
        n=HARMONIC,
        sample=sample,
    )
    np.testing.assert_almost_equal(alpha_eff_bulk, alpha_eff_multi)
