import numpy as np
import matplotlib.pyplot as plt

import finite_dipole as fdm

# Values from Lars Mester Nat. Comms. (2020)
eps_air = 1
eps_PS = 2.5
eps_PMMA = 1.52 + 0.83j
eps_Si = 11.7
eps_stack = eps_air, eps_PS, eps_Si

t_PS = 100e-9
t_PMMA = 100e-9
t_stack = [t_PS,]

beta_stack = fdm.tools.refl_coeff(eps_stack[1:], eps_stack[:1])
beta_q = fdm.multilayer.refl_coeff_ML(beta_stack, t_stack)

z_q = 10e-9
X, beta_X = fdm.multilayer.eff_charge_and_pos(z_q, beta_q)
