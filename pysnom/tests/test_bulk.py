import numpy as np
import pytest

import pysnom


@pytest.mark.parametrize("eff_pol", (pysnom.fdm.eff_pol_bulk, pysnom.pdm.eff_pol_bulk))
def test_eff_pol_broadcasting(eff_pol):
    # Measurement parameters
    wavenumber = np.linspace(1680, 1780, 32) * 1e2
    z = 50e-9
    tapping_amplitude = 50e-9
    harmonic = np.arange(2, 5)[:, np.newaxis]

    # Eventual output shape should match broadcast arrayse
    target_shape = (wavenumber + z + tapping_amplitude + harmonic).shape

    # Dispersive semi-infinite layer dielectric function
    eps_inf = 2
    osc_freq = 1740e2
    osc_width = 20e2
    osc_strength = 15e-3
    eps_sample = eps_inf + (osc_strength * osc_freq**2) / (
        osc_freq**2 - wavenumber**2 - 1j * osc_width * wavenumber
    )

    alpha_eff = eff_pol(
        z=z,
        tapping_amplitude=tapping_amplitude,
        harmonic=harmonic,
        eps_sample=eps_sample,
    )
    assert alpha_eff.shape == target_shape


@pytest.mark.parametrize("eff_pol", (pysnom.fdm.eff_pol_bulk, pysnom.pdm.eff_pol_bulk))
def test_eff_pol_error_if_no_material(eff_pol):
    with pytest.raises(Exception) as e:
        eff_pol(
            z=50e-9,
            tapping_amplitude=50e-9,
            harmonic=np.arange(2, 10),
        )
    assert e.type == ValueError
    assert "Either `eps_sample` or `beta` must be specified." in str(e.value)


@pytest.mark.parametrize("eff_pol", (pysnom.fdm.eff_pol_bulk, pysnom.pdm.eff_pol_bulk))
def test_eff_pol_warning_if_eps_and_beta(eff_pol):
    with pytest.warns(
        UserWarning, match="`beta` overrides `eps_sample` when both are specified."
    ):
        eff_pol(
            z=50e-9,
            tapping_amplitude=50e-9,
            harmonic=np.arange(2, 10),
            eps_sample=2 + 1j,
            beta=0.75,
        )
