import numpy as np
import pytest

import pysnom

eps_env = 1
eps_mid = 2 + 1j
eps_sub = 11.7
t_mid = 50e-9

bulk_sample = pysnom.sample.Sample(eps_stack=(eps_env, eps_sub))
multi_sample = pysnom.sample.Sample(
    eps_stack=(eps_env, eps_mid, eps_sub), t_stack=(t_mid,)
)

demod_kwargs = dict(A_tip=50e-9, n=np.arange(2, 10))


@pytest.mark.parametrize(
    "eff_pol_func, sample, kwargs",
    [
        (pysnom.pdm.eff_pol, bulk_sample, {}),
        (pysnom.pdm.eff_pol_n, bulk_sample, demod_kwargs),
        (pysnom.fdm.bulk.eff_pol, bulk_sample, {}),
        (pysnom.fdm.bulk.eff_pol_n, bulk_sample, demod_kwargs),
        (pysnom.fdm.multi.eff_pol, multi_sample, {}),
        (pysnom.fdm.multi.eff_pol_n, multi_sample, demod_kwargs),
    ],
)
def test_approach_curve_decays(eff_pol_func, sample, kwargs):
    alpha_eff = eff_pol_func(
        z_tip=np.linspace(0, 100, 128)[:, np.newaxis] * 1e-9, sample=sample, **kwargs
    )
    assert (np.diff(np.abs(alpha_eff), axis=0) < 0).all()


@pytest.mark.parametrize(
    "eff_pol_func, sample",
    [
        (pysnom.pdm.eff_pol_n, bulk_sample),
        (pysnom.fdm.bulk.eff_pol_n, bulk_sample),
        (pysnom.fdm.multi.eff_pol_n, multi_sample),
    ],
)
def test_harmonics_decay(eff_pol_func, sample):
    alpha_eff = eff_pol_func(z_tip=10e-9, sample=sample, **demod_kwargs)
    assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()
