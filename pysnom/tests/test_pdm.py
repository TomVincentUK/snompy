import numpy as np
import pytest

import pysnom


def test_eff_pol_warning_if_eps_and_alpha_sphere():
    with pytest.warns(
        UserWarning,
        match="`alpha_sphere` overrides `eps_sphere` when both are specified.",
    ):
        pysnom.pdm.eff_pol(
            z=50e-9,
            tapping_amplitude=50e-9,
            harmonic=np.arange(2, 10),
            beta=0.75,
            eps_sphere=2,
            alpha_sphere=0.75,
        )


def test_eff_pol_uses_perfect_conducting_sphere_if_no_material_specified():
    radius = 20e-9
    alpha_eff_unspecified = pysnom.pdm.eff_pol(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 10),
        beta=0.75,
        radius=radius,
    )
    alpha_eff_perfect = pysnom.pdm.eff_pol(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 10),
        beta=0.75,
        radius=radius,
        alpha_sphere=4 * np.pi * radius**3,
    )
    np.testing.assert_allclose(alpha_eff_unspecified, alpha_eff_perfect)


def test_eff_pol_eps_tip_has_effect():
    alpha_eff_dielectric = pysnom.pdm.eff_pol(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 10),
        beta=0.75,
        eps_sphere=11.7,
    )
    alpha_eff_perfect = pysnom.pdm.eff_pol(
        z=50e-9,
        tapping_amplitude=50e-9,
        harmonic=np.arange(2, 10),
        beta=0.75,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        alpha_eff_dielectric,
        alpha_eff_perfect,
    )
