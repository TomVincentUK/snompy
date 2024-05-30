import warnings

import numpy as np
import pytest
from scipy.integrate import quad_vec

import snompy


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
    nu_vac_None_error = "`nu_vac` must not be None for multilayer samples."
    theta_q_error = "Either `theta_in` or `q` must be None."
    polarization_error = "`polarization` must be 's' or 'p'"

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
            snompy.Sample()

    def test_error_when_both_eps_and_beta(self):
        with pytest.raises(ValueError, match=self.eps_beta_input_error):
            snompy.Sample(eps_stack=(1, 10), beta_stack=(0.5,))

    def test_error_when_eps_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            snompy.Sample(eps_stack=(1, 2, 3, 4, 5), t_stack=(1,))

    def test_error_when_beta_t_incompatible(self):
        with pytest.raises(ValueError, match=self.eps_beta_t_incompatible_error):
            snompy.Sample(beta_stack=(0.5, 0.5, 0.5, 0.5, 0.5), t_stack=(1,))

    def test_transfer_matrix_no_errors_when_bulk_no_nu_vac(self):
        eps_sub = 10
        no_nu_at_init = snompy.bulk_sample(eps_sub=eps_sub)
        no_nu_at_init.refl_coef()

    def test_transfer_matrix_errors_when_multilayer_no_nu_vac(self):
        sample_params = dict(eps_stack=(1, 2, 10), t_stack=(50e-9,))
        nu_vac = 1.0
        # No error when nu_vac specified at init or function call
        nu_at_init = snompy.Sample(nu_vac=nu_vac, **sample_params)
        nu_at_init.refl_coef()

        no_nu_at_init = snompy.Sample(**sample_params)
        no_nu_at_init.refl_coef(nu_vac=nu_vac)

        # Error with no nu_vac
        with pytest.raises(ValueError, match=self.nu_vac_None_error):
            no_nu_at_init.refl_coef()

    def test_transfer_matrix_errors_when_theta_and_q(self):
        sample = snompy.bulk_sample(eps_sub=10, nu_vac=1.0)
        q = theta_in = 0.0

        # No error when theta_in or q specified separately
        sample.transfer_matrix(q=q)
        sample.transfer_matrix(theta_in=theta_in)

        # Error with both
        with pytest.raises(ValueError, match=self.theta_q_error):
            sample.transfer_matrix(q=q, theta_in=theta_in)

    def test_transfer_matrix_errors_for_unknown_polarization(self):
        sample = snompy.bulk_sample(eps_sub=10, nu_vac=1.0)
        # No error for "p" or "s"
        sample.transfer_matrix(polarization="p")
        sample.transfer_matrix(polarization="s")

        # Error with unknown
        with pytest.raises(ValueError, match=self.polarization_error):
            sample.transfer_matrix(polarization="not s or p")

    # Behaviour tests
    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_multilayer_flag(self, eps_stack, beta_stack, t_stack):
        sample = snompy.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        assert sample.multilayer == (np.shape(sample.t_stack)[0] > 0)

    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_eps_beta_conversion_reversible(self, eps_stack, beta_stack, t_stack):
        sample = snompy.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        if eps_stack is not None:
            from_beta = snompy.Sample(beta_stack=sample.beta_stack, t_stack=t_stack)
            np.testing.assert_array_almost_equal(sample.eps_stack, from_beta.eps_stack)
        if beta_stack is not None:
            from_eps = snompy.Sample(eps_stack=sample.eps_stack, t_stack=t_stack)
            np.testing.assert_array_almost_equal(sample.eps_stack, from_eps.eps_stack)

    @pytest.mark.parametrize("eps_i", np.linspace(0.9, 1.1, 3))
    def test_eps_beta_single_conversion_reversible(self, eps_i):
        eps_in = 2 + 1j
        eps_out = snompy.sample.permitivitty(
            beta=snompy.sample.refl_coef_qs_single(eps_i=eps_i, eps_j=eps_in),
            eps_i=eps_i,
        )
        np.testing.assert_allclose(eps_in, eps_out)

        beta_in = 0.5 + 0.5j
        beta_out = snompy.sample.refl_coef_qs_single(
            eps_j=snompy.sample.permitivitty(beta_in, eps_i=eps_i), eps_i=eps_i
        )
        np.testing.assert_allclose(beta_in, beta_out)

    @pytest.mark.parametrize(valid_inputs_kw, valid_inputs)
    def test_outputs_correct_shape(self, eps_stack, beta_stack, t_stack):
        sample = snompy.Sample(
            eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack
        )
        nu_vac = np.ones_like(sample.eps_stack[0], dtype=float)
        assert (
            np.shape(sample.refl_coef_qs())
            == np.shape(sample.trans_coef_qs())
            == np.shape(sample.refl_coef(nu_vac=nu_vac))
            == np.shape(sample.trans_coef(nu_vac=nu_vac))
            == np.shape(sample.refl_coef_qs_above_surf(z_Q=self.z_Q))
            == np.shape(sample.surf_pot_and_field(z_Q=self.z_Q)[0])
            == np.shape(sample.surf_pot_and_field(z_Q=self.z_Q)[1])
            == np.shape(sample.image_depth_and_charge(z_Q=self.z_Q)[0])
            == np.shape(sample.image_depth_and_charge(z_Q=self.z_Q)[1])
            == np.shape(sample.eps_stack[0])
            == np.shape(sample.beta_stack[0])
        )

    def test_refl_coef_qs_flat_for_bulk(self, scalar_sample_bulk):
        q = np.linspace(0, 10, 64)
        beta = scalar_sample_bulk.refl_coef_qs(q)
        np.testing.assert_array_almost_equal(beta - beta.mean(), 0)

    def test_surf_pot_and_field_integrals(self, vector_sample_multi):
        phi, E = vector_sample_multi.surf_pot_and_field(self.z_Q)

        with warnings.catch_warnings():  # scipy quad uses large values that overflow
            warnings.simplefilter("ignore")
            phi_scipy, _ = quad_vec(
                lambda x: vector_sample_multi.refl_coef_qs(x / (2 * self.z_Q))
                * np.exp(-x),
                0,
                np.inf,
            )
            E_scipy, _ = quad_vec(
                lambda x: vector_sample_multi.refl_coef_qs(x / (2 * self.z_Q))
                * x
                * np.exp(-x),
                0,
                np.inf,
            )

        phi_scipy /= 2 * self.z_Q
        phi_valid = ~np.isnan(phi_scipy)
        np.testing.assert_allclose(phi[phi_valid], phi_scipy[phi_valid])

        E_scipy /= 4 * self.z_Q**2
        E_valid = ~np.isnan(E_scipy)
        np.testing.assert_allclose(E[E_valid], E_scipy[E_valid])

    def test_refl_coef_qs_above_surf_integral(self, vector_sample_multi):
        beta_eff = vector_sample_multi.refl_coef_qs_above_surf(self.z_Q)

        with warnings.catch_warnings():  # scipy quad uses large values that overflow
            warnings.simplefilter("ignore")
            numerator_longhand, _ = quad_vec(
                lambda q: vector_sample_multi.refl_coef_qs(q)
                * q
                * np.exp(-2 * self.z_Q * q),
                0,
                np.inf,
            )
            denominator_longhand, _ = quad_vec(
                lambda q: q * np.exp(-2 * self.z_Q * q), 0, np.inf
            )
            beta_eff_scipy, _ = quad_vec(
                lambda x: vector_sample_multi.refl_coef_qs(x / (2 * self.z_Q))
                * x
                * np.exp(-x),
                0,
                np.inf,
            )

        beta_eff_longhand = numerator_longhand / denominator_longhand
        beta_eff_valid = ~np.isnan(beta_eff_longhand + beta_eff_scipy)

        np.testing.assert_allclose(
            beta_eff_scipy[beta_eff_valid], beta_eff_longhand[beta_eff_valid]
        )
        np.testing.assert_allclose(
            beta_eff[beta_eff_valid], beta_eff_scipy[beta_eff_valid]
        )

    def test_image_depth_and_charge_broadcasting(self, vector_sample_multi):
        target_shape = (
            self.z_Q * vector_sample_multi.eps_stack[0] * vector_sample_multi.t_stack[0]
        ).shape
        z_image, beta_image = vector_sample_multi.image_depth_and_charge(self.z_Q)
        assert z_image.shape == beta_image.shape == target_shape

    def test_lorentz_perm_A_j_and_nu_plasma_errors(self):
        with pytest.raises(
            ValueError, match="`A_j` and `nu_plasma` cannot both be None"
        ):
            snompy.sample.lorentz_perm(1, 1, 1)
        with pytest.raises(
            ValueError, match="Either `A_j` or `nu_plasma` must be None"
        ):
            snompy.sample.lorentz_perm(1, 1, 1, A_j=1, nu_plasma=1)

    def test_drude_perm_is_case_of_lorentz_perm(self):
        nu_vac = np.linspace(1000, 2000, 128) * 1e2
        nu_plasma = 10000e2
        gamma = 1000e2
        eps_drude = snompy.sample.drude_perm(nu_vac, nu_plasma=nu_plasma, gamma=gamma)
        eps_lorentz = snompy.sample.lorentz_perm(
            nu_vac, nu_j=0, nu_plasma=nu_plasma, gamma_j=gamma
        )
        np.testing.assert_allclose(eps_drude, eps_lorentz)
