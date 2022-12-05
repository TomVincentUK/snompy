"""
This is not finished.

References
----------
.. [1] Z. M. Zhang, G. Lefever-Button, F. R. Powell,
   Infrared refractive index and extinction coefficient of polyimide films,
   Int. J. Thermophys., 19 (1998) 905.
   https://doi.org/10.1023/A:1022655309574.
.. [2] M.A. Ordal, L.L. Long, R.J. Bell, S.E. Bell, R.R. Bell, R.W.
   Alexander, C.A. Ward,
   Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag,
   Ti, and W in the infrared and far infrared,
   Appl. Opt. 22 (1983) 1099.
   https://doi.org/10.1364/AO.22.001099.
.. [3] Lars Mester Nat. Comms. (2020).
"""
import matplotlib.pyplot as plt
import numpy as np

import finite_dipole as fdm


def eps_Lorentz(omega, eps_inf, omega_0, strength, gamma):
    """
    Lorentzian oscillator dielectric function model. Function definition
    from equation (5) of reference [1]_.
    """
    return eps_inf + (strength * omega_0**2) / (
        omega_0**2 - omega**2 - 1j * gamma * omega
    )


def eps_Drude(omega, eps_inf, omega_plasma, gamma):
    """
    Drude dielectric function model. Function definition from equation (2)
    of reference [2]_.
    """
    return eps_inf - (omega_plasma**2) / (omega**2 + 1j * gamma * omega)


wavenumber = np.linspace(1680, 1780, 129) * 1e2
z_0 = 50e-9
tapping_amplitude = 50e-9
radius = 20e-9
harmonic = 3

# Constant dielectric functions
# Values from [3]_
eps_air = 1
eps_Si = 11.7

# Dispersive dielectric functions
# (My simplified model for the PMMA C=O bond based on fig 5a of [3]_)
eps_PMMA = eps_Lorentz(wavenumber, 2, 1738e2, 14e-2, 20e2)
eps_Au = eps_Drude(wavenumber, 1, 7.25e6, 2.16e4)  # values from [2]_

eps_stack = np.broadcast_arrays(eps_air, eps_PMMA, eps_Si)

t_PMMA = 100e-9
t_stack = (t_PMMA,)

beta_stack = fdm.tools.refl_coeff(eps_stack[:-1], eps_stack[1:])
beta_k = [fdm.multilayer.refl_coeff_ML(b, t_stack) for b in beta_stack.T]
