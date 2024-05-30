import numpy as np
import pytest

import snompy


class TestEffPol:
    model_and_kwargs = [
        (snompy.fdm, {"method": "bulk"}),
        (snompy.fdm, {"method": "multi"}),
        (snompy.fdm, {"method": "Q_ave"}),
        (snompy.pdm, {}),
    ]
    taylor_model_and_kwargs = [
        (snompy.fdm, {"method": "bulk"}),
        (snompy.fdm, {"method": "Q_ave"}),
    ]
    inverse_model = [snompy.fdm, snompy.pdm]

    # eff_pol

    @pytest.mark.parametrize("model, model_kwargs", model_and_kwargs)
    def test_approach_curve_decays(self, model, model_kwargs, scalar_sample_bulk):
        alpha_eff = model.eff_pol(
            sample=scalar_sample_bulk,
            **dict(z_tip=np.linspace(1, 100, 32) * 1e-9) | model_kwargs
        )
        assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()

    @pytest.mark.parametrize("model, model_kwargs", model_and_kwargs)
    def test_broadcasting(
        self, model, model_kwargs, vector_sample_bulk, vector_AFM_params
    ):
        a_sum = 0
        for a in [vector_sample_bulk.eps_stack[0], *vector_AFM_params.values()]:
            a_sum = a_sum + np.asarray(a)
        target_shape = a_sum.shape

        alpha_eff = model.eff_pol(
            sample=vector_sample_bulk, **vector_AFM_params | model_kwargs
        )

        assert alpha_eff.shape == target_shape

    # eff_pol_n

    @pytest.mark.parametrize("model, model_kwargs", model_and_kwargs)
    def test_approach_curve_n_decays(
        self, model, model_kwargs, scalar_sample_bulk, scalar_tapping_params
    ):
        alpha_eff = model.eff_pol_n(
            sample=scalar_sample_bulk,
            **scalar_tapping_params
            | dict(z_tip=np.linspace(1, 100, 32) * 1e-9)
            | model_kwargs
        )
        assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()

    @pytest.mark.parametrize("model, model_kwargs", model_and_kwargs)
    def test_harmonics_decay(
        self, model, model_kwargs, scalar_sample_bulk, scalar_tapping_params
    ):
        alpha_eff = model.eff_pol_n(
            sample=scalar_sample_bulk,
            **scalar_tapping_params | dict(n=np.arange(2, 10)) | model_kwargs
        )
        assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()

    @pytest.mark.parametrize("model, model_kwargs", model_and_kwargs)
    def test_demod_broadcasting(
        self,
        model,
        model_kwargs,
        vector_sample_bulk,
        vector_AFM_params,
        vector_tapping_params,
    ):
        a_sum = 0
        for a in [
            vector_sample_bulk.eps_stack[0],
            *(vector_AFM_params | vector_tapping_params).values(),
        ]:
            a_sum = a_sum + np.asarray(a)
        target_shape = a_sum.shape

        alpha_eff = model.eff_pol_n(
            sample=vector_sample_bulk,
            **vector_AFM_params | vector_tapping_params | model_kwargs
        )

        assert alpha_eff.shape == target_shape

    # Multilayer

    def test_multi_same_as_bulk_for_single_interface(
        self, vector_sample_bulk, vector_AFM_params, vector_tapping_params
    ):
        params = dict(
            sample=vector_sample_bulk, **vector_AFM_params | vector_tapping_params
        )
        np.testing.assert_allclose(
            snompy.fdm.eff_pol_n(method="bulk", **params),
            snompy.fdm.eff_pol_n(method="multi", **params),
            rtol=1e-4,
        )

    # Taylor series

    @pytest.mark.parametrize("model, model_kwargs", taylor_model_and_kwargs)
    def test_approach_curve_n_taylor_decays(
        self, model, model_kwargs, scalar_sample_bulk, scalar_tapping_params
    ):
        alpha_eff = model.eff_pol_n_taylor(
            sample=scalar_sample_bulk,
            **scalar_tapping_params
            | dict(z_tip=np.linspace(1, 100, 32) * 1e-9)
            | model_kwargs
        )
        assert (np.diff(np.abs(alpha_eff), axis=-1) < 0).all()

    @pytest.mark.parametrize("model, model_kwargs", taylor_model_and_kwargs)
    def test_taylor_same_as_bulk(
        self,
        model,
        model_kwargs,
        vector_sample_bulk,
        vector_AFM_params,
        vector_tapping_params,
    ):
        params = dict(
            sample=vector_sample_bulk,
            **vector_AFM_params | vector_tapping_params | model_kwargs
        )
        np.testing.assert_allclose(
            model.eff_pol_n(**params), model.eff_pol_n_taylor(**params), rtol=1e-3
        )

    @pytest.mark.parametrize("model, model_kwargs", taylor_model_and_kwargs)
    def test_taylor_broadcasting(
        self,
        model,
        model_kwargs,
        vector_sample_bulk,
        vector_AFM_params,
        vector_tapping_params,
    ):
        a_sum = 0
        for a in [
            vector_sample_bulk.eps_stack[0],
            *(vector_AFM_params | vector_tapping_params).values(),
        ]:
            a_sum = a_sum + np.asarray(a)
        target_shape = a_sum.shape

        alpha_eff = model.eff_pol_n_taylor(
            sample=vector_sample_bulk,
            **vector_AFM_params | vector_tapping_params | model_kwargs
        )

        assert alpha_eff.shape == target_shape

    # Inverse functions

    @pytest.mark.parametrize("model", inverse_model)
    def test_refl_coef_qs_from_eff_pol(self, model, vector_AFM_params):
        n_test_beta = 10
        beta_in = np.linspace(0.9, 0.1, n_test_beta) * np.exp(
            1j * np.linspace(0, np.pi, n_test_beta)
        )
        sample = snompy.bulk_sample(beta=beta_in)

        alpha_eff = model.eff_pol(sample=sample, **vector_AFM_params)
        beta_out = model.refl_coef_qs_from_eff_pol(
            alpha_eff=alpha_eff, **vector_AFM_params
        )

        atol = 1e-12
        assert np.all(np.abs(beta_out - beta_in) < atol)

    @pytest.mark.parametrize("model", inverse_model)
    def test_refl_coef_qs_from_eff_pol_n(
        self, model, vector_AFM_params, vector_tapping_params
    ):
        n_test_beta = 10
        beta_in = np.linspace(0.9, 0.1, n_test_beta) * np.exp(
            1j * np.linspace(0, np.pi, n_test_beta)
        )
        beta_in = np.hstack([beta_in, -0.5 + 0.5j])  # a case with multiple solutions
        sample = snompy.bulk_sample(beta=beta_in)

        alpha_eff_n = model.eff_pol_n_taylor(
            sample=sample, **vector_AFM_params | vector_tapping_params
        )
        beta_out = model.refl_coef_qs_from_eff_pol_n(
            alpha_eff_n=alpha_eff_n, **vector_AFM_params | vector_tapping_params
        )

        # beta_out may contain multiple solutions
        # need to check if any solutions correspond to the input
        atol = 0
        rtol = 1.0e-7
        close_values = np.abs(beta_out - beta_in) <= (atol + rtol * np.abs(beta_in))
        assert close_values.any(axis=0).all()

    @pytest.mark.parametrize("model", inverse_model)
    def test_refl_coef_qs_from_eff_pol_n_reject_negative_eps_imag(
        self, model, vector_AFM_params, vector_tapping_params
    ):
        eps_in = np.array([10 + 1j, 10 - 1j])  # 1 valid, 1 invalid
        sample = snompy.bulk_sample(eps_in)

        alpha_eff_n = model.eff_pol_n_taylor(
            sample=sample, **vector_AFM_params | vector_tapping_params
        )
        beta_out = model.refl_coef_qs_from_eff_pol_n(
            alpha_eff_n=alpha_eff_n,
            reject_negative_eps_imag=True,
            **vector_AFM_params | vector_tapping_params
        )
        eps_out = snompy.sample.permitivitty(beta_out)

        assert (eps_out.imag >= 0).all()

    @pytest.mark.parametrize("model", inverse_model)
    def test_refl_coef_qs_from_eff_pol_n_reject_subvacuum_eps_abs(
        self, model, vector_AFM_params, vector_tapping_params
    ):
        eps_in = np.array([10 + 10j, 0.1 + 0.1j])  # 1 valid, 1 invalid
        sample = snompy.bulk_sample(eps_in)

        alpha_eff_n = model.eff_pol_n_taylor(
            sample=sample, **vector_AFM_params | vector_tapping_params
        )
        beta_out = model.refl_coef_qs_from_eff_pol_n(
            alpha_eff_n=alpha_eff_n,
            reject_subvacuum_eps_abs=True,
            **vector_AFM_params | vector_tapping_params
        )
        eps_out = snompy.sample.permitivitty(beta_out)

        assert (np.abs(eps_out) >= 1).all()
