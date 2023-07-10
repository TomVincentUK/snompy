import numpy as np
import pytest

from pysnom.sample import Sample


class TestSample:
    # Error messages
    eps_beta_input_error = r" ".join(
        [
            r"Either `eps_stack` or `beta_stack` must be specified,",
            r"\(but not both\).",
        ]
    )

    eps_beta_t_incompatible_error = " ".join(
        [
            "Invalid inputs:",
            "`eps_stack` must be 2 longer than `t_stack`, or",
            "`beta_stack` must be 1 longer than `t_stack`,",
            "along the first axis.",
        ]
    )

    valid_inputs_kw = ", ".join(["eps_stack", "beta_stack", "t_stack"])
    valid_inputs = (
        ([1, 2], None, []),
        ([1, 2], None, None),
        ([1, 2, 3], None, [10e-9]),
        ([1, 2, 3, 4], None, [10e-9, 20e-9]),
        ([[1], [2]], None, []),
        ([[1, 1], [2, 3]], None, []),
        (None, [1 / 2], []),
        (None, [1 / 2], None),
        (None, [1 / 2, 1 / 3], [10e-9]),
        (None, [1 / 2, 1 / 3, 1 / 4], [10e-9, 20e-9]),
        (None, [[1 / 2]], []),
        (None, [[1 / 2, 1 / 3]], []),
        (None, [[1 / 2, 1 / 2], [1 / 3, 1 / 4]], [10e-9]),
    )

    # Input tests
    def test_error_when_no_eps_or_beta(self):
        with pytest.raises(ValueError, match=self.eps_beta_input_error):
            Sample()

    def test_error_when_both_eps_and_beta(self):
        with pytest.raises(ValueError, match=self.eps_beta_input_error):
            Sample(eps_stack=(1, 10), beta_stack=(0.5,))

    def test_error_when_eps_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            Sample(eps_stack=(1, 2, 3, 4, 5), t_stack=(1,))

    def test_error_when_beta_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            Sample(beta_stack=(0.5, 0.5, 0.5, 0.5, 0.5), t_stack=(1,))

    def test_warn_when_zero_thickness(self):
        with pytest.warns(
            UserWarning,
            match=" ".join(
                [
                    "`t_stack` contains zeros.",
                    "Zero-thickness dielectric layers are unphysical.",
                    "Results may not be as expected.",
                ]
            ),
        ):
            Sample(eps_stack=(1, 2, 3, 4, 5), t_stack=(1, 0, 1))

    # Behaviour tests
    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_eps_beta_conversion_reversible(self, eps_stack, beta_stack, t_stack):
        sample = Sample(eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack)
        if eps_stack is not None:
            from_beta = Sample(beta_stack=sample.beta_stack, t_stack=t_stack)
            np.testing.assert_almost_equal(sample.eps_stack, from_beta.eps_stack)
        if beta_stack is not None:
            from_eps = Sample(eps_stack=sample.eps_stack, t_stack=t_stack)
            np.testing.assert_almost_equal(sample.eps_stack, from_eps.eps_stack)

    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_multilayer_flag(self, eps_stack, beta_stack, t_stack):
        sample = Sample(eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack)
        assert sample.multilayer == (np.shape(sample.t_stack)[0] > 0)

    # def test_expected_shape_from_inputs(self):
    #     pass
