from .finite_dipole import (
    geom_func,
    eff_pol_0,
    eff_pol,
)
from . import tools
from . import _version

__all__ = [geom_func, eff_pol_0, eff_pol, tools]

__version__ = _version.get_versions()["version"]
