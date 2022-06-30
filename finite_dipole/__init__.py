from .finite_dipole import (
    complex_quad,
    refl_coeff,
    geom_func,
    eff_pol_0,
    Fourier_envelope,
    eff_pol,
)

__all__ = [
    complex_quad,
    refl_coeff,
    geom_func,
    eff_pol_0,
    Fourier_envelope,
    eff_pol,
]

from . import _version

__version__ = _version.get_versions()["version"]
