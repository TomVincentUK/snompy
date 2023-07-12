import numpy as np
import pytest

import pysnom


class TestPDM:
    def test_eff_pol_warning_if_eps_and_alpha_sphere(
        self, scalar_sample_bulk, scalar_AFM_params, scalar_tapping_params
    ):
        with pytest.raises(
            ValueError, match="Either `alpha_sphere` or `eps_sphere` must be None."
        ):
            pysnom.pdm.eff_pol_n(
                eps_sphere=2,
                alpha_sphere=0.75,
                sample=scalar_sample_bulk,
                **scalar_AFM_params | scalar_tapping_params
            )

    def test_eff_pol_uses_perfect_conducting_sphere_if_no_material_specified(
        self, scalar_sample_bulk, scalar_AFM_params, scalar_tapping_params
    ):
        alpha_eff_unspecified = pysnom.pdm.eff_pol_n(
            sample=scalar_sample_bulk, **scalar_AFM_params, **scalar_tapping_params
        )
        alpha_eff_perfect = pysnom.pdm.eff_pol_n(
            alpha_sphere=4 * np.pi * scalar_AFM_params["r_tip"] ** 3,
            sample=scalar_sample_bulk,
            **scalar_AFM_params | scalar_tapping_params
        )
        np.testing.assert_allclose(alpha_eff_unspecified, alpha_eff_perfect)

    def test_eff_pol_eps_sphere_has_effect(
        self, scalar_sample_bulk, scalar_AFM_params, scalar_tapping_params
    ):
        alpha_eff_perfect = pysnom.pdm.eff_pol_n(
            sample=scalar_sample_bulk, **scalar_AFM_params | scalar_tapping_params
        )
        alpha_eff_dielectric = pysnom.pdm.eff_pol_n(
            eps_sphere=10,
            sample=scalar_sample_bulk,
            **scalar_AFM_params | scalar_tapping_params
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            alpha_eff_dielectric,
            alpha_eff_perfect,
        )
