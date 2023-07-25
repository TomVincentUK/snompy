import pytest

import pysnom


class TestFDM:
    # Error messages
    multilayer_sample_for_bulk_error = (
        "`method`='bulk' cannot be used for multilayer samples."
    )
    unknown_method_error = "`method` must be one of `bulk`, `Hauer`, or `Mester`."
    unknown_method_taylor_error = "`method` must be one of `bulk`, or `Mester`."

    def test_eff_pol_error_bulk_used_for_multilayer_sample(
        self, scalar_sample_multi, scalar_AFM_params
    ):
        with pytest.raises(ValueError, match=self.multilayer_sample_for_bulk_error):
            pysnom.fdm.eff_pol(
                sample=scalar_sample_multi, **scalar_AFM_params, method="bulk"
            )

    def test_eff_pol_n_taylor_error_bulk_used_for_multilayer_sample(
        self, scalar_sample_multi, scalar_AFM_params, scalar_tapping_params
    ):
        with pytest.raises(ValueError, match=self.multilayer_sample_for_bulk_error):
            pysnom.fdm.eff_pol_n_taylor(
                sample=scalar_sample_multi,
                **scalar_AFM_params | scalar_tapping_params,
                method="bulk"
            )

    def test_eff_pol_error_for_unknown_method(
        self, scalar_sample_multi, scalar_AFM_params
    ):
        with pytest.raises(ValueError, match=self.unknown_method_error):
            pysnom.fdm.eff_pol(
                sample=scalar_sample_multi, **scalar_AFM_params, method="not a method"
            )

    def test_eff_pol_n_taylor_error_for_unknown_method(
        self, scalar_sample_multi, scalar_AFM_params, scalar_tapping_params
    ):
        with pytest.raises(ValueError, match=self.unknown_method_taylor_error):
            pysnom.fdm.eff_pol_n_taylor(
                sample=scalar_sample_multi,
                **scalar_AFM_params | scalar_tapping_params,
                method="not a method"
            )
