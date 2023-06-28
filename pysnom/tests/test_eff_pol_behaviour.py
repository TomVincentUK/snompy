import numpy as np
import pytest

import pysnom

eps_env = 1
eps_middle = 2 + 1j
eps_substrate = 11.7
t_middle = 50e-9

beta = pysnom.reflection.refl_coef_qs(eps_env, eps_substrate)
beta_stack, t_stack = pysnom.reflection.interface_stack(
    eps_stack=(eps_env, eps_middle, eps_substrate), t_stack=(t_middle,)
)

bulk_kwargs = dict(beta=beta)
ML_kwargs = dict(beta_stack=beta_stack, t_stack=t_stack)
demod_kwargs = dict(A_tip=50e-9, n=np.arange(2, 10))
funcs_0_kwargs = [
    (pysnom.pdm.eff_pol, bulk_kwargs),
    (pysnom.fdm.bulk.eff_pol, bulk_kwargs),
    (pysnom.fdm.multi.eff_pol, ML_kwargs),
]
funcs_demod_kwargs = [
    (pysnom.pdm.eff_pol_n, bulk_kwargs | demod_kwargs),
    (pysnom.fdm.bulk.eff_pol_n, bulk_kwargs | demod_kwargs),
    (pysnom.fdm.multi.eff_pol_n, ML_kwargs | demod_kwargs),
]


@pytest.mark.parametrize("eff_pol_func, kwargs", funcs_0_kwargs + funcs_demod_kwargs)
def test_approach_curve_decays(eff_pol_func, kwargs):
    alpha_eff = eff_pol_func(
        z_tip=np.linspace(0, 100, 128)[:, np.newaxis] * 1e-9, **kwargs
    )
    assert (np.diff(np.abs(alpha_eff), axis=0) < 0).all()


@pytest.mark.parametrize("eff_pol_func, kwargs", funcs_demod_kwargs)
def test_harmonics_decay(eff_pol_func, kwargs):
    alpha_eff = eff_pol_func(z_tip=10e-9, **kwargs)
    assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()
