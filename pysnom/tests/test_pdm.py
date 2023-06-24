import numpy as np
import pytest

import pysnom


def test_eff_pol_warning_if_eps_and_alpha_sphere():
    with pytest.warns(
        UserWarning,
        match="`alpha_sphere` overrides `eps_sphere` when both are specified.",
    ):
        pysnom.pdm.eff_pol_n(
            z_tip=50e-9,
            A_tip=50e-9,
            n=np.arange(2, 10),
            beta=0.75,
            eps_sphere=2,
            alpha_sphere=0.75,
        )


def test_eff_pol_uses_perfect_conducting_sphere_if_no_material_specified():
    r_tip = 20e-9
    alpha_eff_unspecified = pysnom.pdm.eff_pol_n(
        z_tip=50e-9,
        A_tip=50e-9,
        n=np.arange(2, 10),
        beta=0.75,
        r_tip=r_tip,
    )
    alpha_eff_perfect = pysnom.pdm.eff_pol_n(
        z_tip=50e-9,
        A_tip=50e-9,
        n=np.arange(2, 10),
        beta=0.75,
        r_tip=r_tip,
        alpha_sphere=4 * np.pi * r_tip**3,
    )
    np.testing.assert_allclose(alpha_eff_unspecified, alpha_eff_perfect)


def test_eff_pol_eps_tip_has_effect():
    alpha_eff_dielectric = pysnom.pdm.eff_pol_n(
        z_tip=50e-9,
        A_tip=50e-9,
        n=np.arange(2, 10),
        beta=0.75,
        eps_sphere=11.7,
    )
    alpha_eff_perfect = pysnom.pdm.eff_pol_n(
        z_tip=50e-9,
        A_tip=50e-9,
        n=np.arange(2, 10),
        beta=0.75,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        alpha_eff_dielectric,
        alpha_eff_perfect,
    )
