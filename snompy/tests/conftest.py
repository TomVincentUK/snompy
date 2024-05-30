import numpy as np
import pytest

import snompy


@pytest.fixture
def scalar_sample_bulk():
    return snompy.bulk_sample(2 + 1j)


@pytest.fixture
def scalar_sample_multi():
    return snompy.Sample(eps_stack=(1, 1.5, 2 + 1j, 10), t_stack=(20e-9, 50e-9))


@pytest.fixture
def vector_sample_bulk():
    # Dispersive medium
    eps_substrate = snompy.sample.lorentz_perm(
        nu_vac=np.linspace(1680, 1780, 128) * 1e2,
        nu_j=1738e2,
        gamma_j=20e2,
        A_j=14e-3,
        eps_inf=2,
    )
    return snompy.bulk_sample(eps_substrate)


@pytest.fixture
def vector_sample_multi():
    # Dispersive medium and different thicknesses
    eps_env = 1
    eps_top = 1.5
    eps_substrate = 11.7
    eps_middle = snompy.sample.lorentz_perm(
        nu_vac=np.linspace(1680, 1780, 128) * 1e2,
        nu_j=1738e2,
        gamma_j=20e2,
        A_j=14e-3,
        eps_inf=2,
    )
    t_top = 20e-9
    t_middle = np.geomspace(10, 100, 32)[:, np.newaxis] * 1e-9
    return snompy.Sample(
        eps_stack=(eps_env, eps_top, eps_middle, eps_substrate),
        t_stack=(t_top, t_middle),
    )


@pytest.fixture
def vector_AFM_params():
    return dict(z_tip=np.linspace(0, 100, 32)[:, np.newaxis, np.newaxis] * 1e-9)


@pytest.fixture
def scalar_tapping_params():
    return dict(A_tip=25e-9, n=3)


@pytest.fixture
def vector_tapping_params(scalar_tapping_params):
    return scalar_tapping_params | dict(
        n=np.arange(2, 10)[:, np.newaxis, np.newaxis, np.newaxis]
    )
