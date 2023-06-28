import numpy as np
import pytest

import pysnom


@pytest.mark.parametrize(
    "eff_pol_n",
    (
        pysnom.fdm.bulk.eff_pol_n,
        pysnom.fdm.bulk.eff_pol_n_taylor,
        pysnom.pdm.eff_pol_n,
    ),
)
def test_eff_pol_n_broadcasting(eff_pol_n):
    # Measurement parameters
    wavenumber = np.linspace(1680, 1780, 32) * 1e2
    z_tip = 50e-9
    A_tip = 50e-9
    n = np.arange(2, 5)[:, np.newaxis]

    # Eventual output shape should match broadcast arrays
    target_shape = (wavenumber + z_tip + A_tip + n).shape

    # Dispersive semi-infinite layer dielectric function
    eps_inf = 2
    osc_freq = 1740e2
    osc_width = 20e2
    osc_strength = 15e-3
    eps_samp = eps_inf + (osc_strength * osc_freq**2) / (
        osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
    )

    alpha_eff = eff_pol_n(
        z_tip=z_tip,
        A_tip=A_tip,
        n=n,
        eps_samp=eps_samp,
    )
    assert alpha_eff.shape == target_shape


@pytest.mark.parametrize(
    "eff_pol_n",
    (
        pysnom.fdm.bulk.eff_pol_n,
        pysnom.fdm.bulk.eff_pol_n_taylor,
        pysnom.pdm.eff_pol_n,
    ),
)
def test_eff_pol_n_error_if_no_material(eff_pol_n):
    with pytest.raises(Exception) as e:
        eff_pol_n(
            z_tip=50e-9,
            A_tip=50e-9,
            n=np.arange(2, 10),
        )
    assert e.type == ValueError
    assert "Either `eps_samp` or `beta` must be specified." in str(e.value)


@pytest.mark.parametrize(
    "eff_pol_n",
    (
        pysnom.fdm.bulk.eff_pol_n,
        pysnom.fdm.bulk.eff_pol_n_taylor,
        pysnom.pdm.eff_pol_n,
    ),
)
def test_eff_pol_n_warning_if_eps_and_beta(eff_pol_n):
    with pytest.warns(
        UserWarning, match="`beta` overrides `eps_samp` when both are specified."
    ):
        eff_pol_n(
            z_tip=50e-9,
            A_tip=50e-9,
            n=np.arange(2, 10),
            eps_samp=2 + 1j,
            beta=0.75,
        )


@pytest.mark.parametrize("model", (pysnom.fdm.bulk,))
def test_eff_pol_n_taylor_equals_eff_pol_n(model):
    n_test_beta = 10
    beta = np.linspace(1, 0.1, n_test_beta) * np.exp(
        1j * np.linspace(0, np.pi, n_test_beta)
    )
    params = dict(
        z_tip=50e-9,
        A_tip=50e-9,
        n=3,
        beta=beta,
    )

    np.testing.assert_allclose(
        model.eff_pol_n(**params), model.eff_pol_n_taylor(**params)
    )


@pytest.mark.parametrize("model", (pysnom.fdm.bulk,))
def test_refl_coef_qs_from_eff_pol_n_bulk_taylor(model):
    n_test_beta = 10
    beta_in = np.linspace(1, 0.1, n_test_beta) * np.exp(
        1j * np.linspace(0, np.pi, n_test_beta)
    )
    # Add a case with multiple solutions
    beta_in = np.hstack([beta_in, -0.5 + 0.5j])
    params = dict(z_tip=1e-9, A_tip=30e-9, n=np.arange(2, 6)[:, np.newaxis])
    alpha_eff_n = model.eff_pol_n_taylor(beta=beta_in, **params)
    beta_out = model.refl_coef_qs_from_eff_pol_n(alpha_eff_n=alpha_eff_n, **params)

    # beta_out may contain multiple solutions
    # need to check if any solutions correspond to the input
    atol = 0
    rtol = 1.0e-7
    close_values = np.abs(beta_out - beta_in) <= (atol + rtol * np.abs(beta_in))
    assert close_values.any(axis=0).all()
