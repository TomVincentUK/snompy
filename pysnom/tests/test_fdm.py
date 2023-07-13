import numpy as np
from scipy.integrate import quad_vec

import pysnom


class TestFDM:
    z_Q = 60e-9

    def test_phi_E_0_integrals(self, vector_sample_multi):
        phi, E = pysnom.fdm.multi.phi_E_0(self.z_Q, vector_sample_multi)

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

    def test_eff_pos_and_charge_broadcasting(self, vector_sample_multi):
        target_shape = (
            self.z_Q * vector_sample_multi.eps_stack[0] * vector_sample_multi.t_stack[0]
        ).shape
        z_image, beta_image = pysnom.fdm.multi.eff_pos_and_charge(
            self.z_Q, vector_sample_multi
        )
        assert z_image.shape == beta_image.shape == target_shape
