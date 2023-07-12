import numpy as np
import pytest

import pysnom


@pytest.fixture
def scalar_sample_bulk():
    return pysnom.sample.bulk_sample(2 + 1j)


@pytest.fixture
def scalar_sample_multi():
    return pysnom.sample.Sample(eps_stack=(1, 2 + 1j, 10), t_stack=(50e-9,))


@pytest.fixture
def vector_sample_bulk():
    # Dispersive medium
    eps_inf = 2
    osc_freq = 1738e2
    osc_width = 20e2
    osc_strength = 14e-3
    wavenumber = np.linspace(1680, 1780, 128) * 1e2
    eps_substrate = eps_inf + (osc_strength * osc_freq**2) / (
        osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
    )  # Lorentzian oscillator
    return pysnom.sample.bulk_sample(eps_substrate)


@pytest.fixture
def vector_sample_multi():
    # Dispersive medium and different thicknesses
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
    return pysnom.sample.Sample(
        eps_stack=(eps_env, eps_middle, eps_substrate), t_stack=(thickness,)
    )


@pytest.fixture
def scalar_AFM_params():
    return dict(z_tip=1e-9, r_tip=20e-9)


@pytest.fixture
def vector_AFM_params(scalar_AFM_params):
    return scalar_AFM_params | dict(
        z_tip=np.linspace(1, 100, 32)[:, np.newaxis, np.newaxis] * 1e-9
    )


@pytest.fixture
def scalar_tapping_params():
    return dict(A_tip=25e-9, n=3)


@pytest.fixture
def vector_tapping_params(scalar_tapping_params):
    return scalar_tapping_params | dict(
        n=np.arange(2, 10)[:, np.newaxis, np.newaxis, np.newaxis]
    )
