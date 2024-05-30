import numpy as np
import pytest

import snompy


class TestPDM:
    def test_eff_pol_error_if_eps_and_alpha_tip(
        self, scalar_sample_bulk, scalar_tapping_params
    ):
        with pytest.raises(
            ValueError, match="Either `alpha_tip` or `eps_tip` must be None."
        ):
            snompy.pdm.eff_pol_n(
                eps_tip=2,
                alpha_tip=0.75,
                sample=scalar_sample_bulk,
                **scalar_tapping_params
            )

    def test_eff_pol_uses_perfect_conducting_sphere_if_no_material_specified(
        self, scalar_sample_bulk, scalar_tapping_params
    ):
        r_tip = 20e-9
        alpha_eff_unspecified = snompy.pdm.eff_pol_n(
            sample=scalar_sample_bulk, r_tip=r_tip, **scalar_tapping_params
        )
        alpha_eff_perfect = snompy.pdm.eff_pol_n(
            alpha_tip=4 * np.pi * r_tip**3,
            sample=scalar_sample_bulk,
            **scalar_tapping_params
        )
        np.testing.assert_allclose(alpha_eff_unspecified, alpha_eff_perfect)

    def test_eff_pol_eps_tip_has_effect(
        self, scalar_sample_bulk, scalar_tapping_params
    ):
        alpha_eff_perfect = snompy.pdm.eff_pol_n(
            sample=scalar_sample_bulk, **scalar_tapping_params
        )
        alpha_eff_dielectric = snompy.pdm.eff_pol_n(
            eps_tip=10, sample=scalar_sample_bulk, **scalar_tapping_params
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            alpha_eff_dielectric,
            alpha_eff_perfect,
        )
