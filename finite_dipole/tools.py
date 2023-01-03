"""
General tools used by other modules.

References
----------
.. [1] B. Hauer, A.P. Engelhardt, T. Taubner,
   Quasi-analytical model for scattering infrared near-field microscopy on
   layered systems,
   Opt. Express. 20 (2012) 13173.
   https://doi.org/10.1364/OE.20.013173.
"""
from numba import vectorize


@vectorize(["float64(float64, float64)", "complex128(complex128, complex128)"])
def refl_coeff(eps_i, eps_j):
    """
    Electrostatic reflection coefficient for an interface between materials
    i and j. Defined as :math:`\beta_{ij}`` in equation (7) of reference
    [1]_.

    Parameters
    ----------
    eps_i : complex
        Dielectric function of material i.
    eps_j : complex, default 1 + 0j
        Dielectric function of material j.

    Returns
    -------
    beta_ij : complex
        Electrostatic reflection coefficient of the sample.
    """
    return (eps_j - eps_i) / (eps_j + eps_i)
