import numpy as np
import pytest
from scipy.integrate import quad_vec

import pysnom


class TestSample:
    # Error messages
    eps_beta_input_error = r" ".join(
        [r"Either `eps_stack` or `beta_stack` must be specified,", r"\(but not both\)."]
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

    z_Q = 60e-9

    # Input tests
    def test_error_when_no_eps_or_beta(self):
        with pytest.raises(ValueError, match=self.eps_beta_input_error):
            pysnom.sample.Sample()

    def test_error_when_both_eps_and_beta(self):
        with pytest.raises(ValueError, match=self.eps_beta_input_error):
            pysnom.sample.Sample(eps_stack=(1, 10), beta_stack=(0.5,))

    def test_error_when_eps_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            pysnom.sample.Sample(eps_stack=(1, 2, 3, 4, 5), t_stack=(1,))

    def test_error_when_beta_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            pysnom.sample.Sample(beta_stack=(0.5, 0.5, 0.5, 0.5, 0.5), t_stack=(1,))

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
            pysnom.sample.Sample(eps_stack=(1, 2, 3, 4, 5), t_stack=(1, 0, 1))

    # Behaviour tests
    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_multilayer_flag(self, eps_stack, beta_stack, t_stack):
        sample = pysnom.sample.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        assert sample.multilayer == (np.shape(sample.t_stack)[0] > 0)

    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_eps_beta_conversion_reversible(self, eps_stack, beta_stack, t_stack):
        sample = pysnom.sample.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        if eps_stack is not None:
            from_beta = pysnom.sample.Sample(
                beta_stack=sample.beta_stack, t_stack=t_stack
            )
            np.testing.assert_array_almost_equal(sample.eps_stack, from_beta.eps_stack)
        if beta_stack is not None:
            from_eps = pysnom.sample.Sample(eps_stack=sample.eps_stack, t_stack=t_stack)
            np.testing.assert_array_almost_equal(sample.eps_stack, from_eps.eps_stack)

    @pytest.mark.parametrize("eps_i", np.linspace(0.9, 1.1, 3))
    def test_eps_beta_single_conversion_reversible(self, eps_i):
        eps_in = 2 + 1j
        eps_out = pysnom.sample.permitivitty(
            beta=pysnom.sample.refl_coef_qs_single(eps_i=eps_i, eps_j=eps_in),
            eps_i=eps_i,
        )
        np.testing.assert_allclose(eps_in, eps_out)

        beta_in = 0.5 + 0.5j
        beta_out = pysnom.sample.refl_coef_qs_single(
            eps_j=pysnom.sample.permitivitty(beta_in, eps_i=eps_i), eps_i=eps_i
        )
        np.testing.assert_allclose(beta_in, beta_out)

    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_outputs_correct_shape(self, eps_stack, beta_stack, t_stack):
        sample = pysnom.sample.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        assert (
            np.shape(sample.refl_coef_qs())
            == np.shape(sample.eps_stack[0])
            == np.shape(sample.beta_stack[0])
        )

    def test_refl_coeff_qs_flat_for_bulk(self, scalar_sample_bulk):
        q = np.linspace(0, 10, 64)
        beta = scalar_sample_bulk.refl_coef_qs(q)
        print(beta - beta.mean())
        np.testing.assert_array_almost_equal(beta - beta.mean(), 0)

    def test_surf_pot_and_field_integrals(self, vector_sample_multi):
        phi, E = vector_sample_multi.surf_pot_and_field(self.z_Q)

        phi_scipy, _ = quad_vec(
            lambda x: vector_sample_multi.refl_coef_qs(x / (2 * self.z_Q)) * np.exp(-x),
            0,
            np.inf,
        )
        phi_scipy /= 2 * self.z_Q
        np.testing.assert_allclose(phi, phi_scipy)

        E_scipy, _ = quad_vec(
            lambda x: vector_sample_multi.refl_coef_qs(x / (2 * self.z_Q))
            * x
            * np.exp(-x),
            0,
            np.inf,
        )
        E_scipy /= 4 * self.z_Q**2
        np.testing.assert_allclose(E, E_scipy)

    def test_image_depth_and_charge_broadcasting(self, vector_sample_multi):
        target_shape = (
            self.z_Q * vector_sample_multi.eps_stack[0] * vector_sample_multi.t_stack[0]
        ).shape
        z_image, beta_image = vector_sample_multi.image_depth_and_charge(self.z_Q)
        assert z_image.shape == beta_image.shape == target_shape
