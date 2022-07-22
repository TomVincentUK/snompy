import numpy as np
import matplotlib.pyplot as plt

import finite_dipole as fdm

# Values from Lars Mester Nat. Comms. (2020)
eps_air = 1
eps_PS = 2.5
eps_PMMA = 1.52 + 0.83j
eps_Si = 11.7
eps_stack = eps_air, eps_PMMA, eps_Si

t_PS = 100e-9
t_PMMA = 100e-9
t_stack = (t_PMMA,)

beta_stack = fdm.tools.refl_coeff(eps_stack[:1], eps_stack[1:])
beta_k = fdm.multilayer.refl_coeff_ML(beta_stack, t_stack)

z_0 = np.linspace(0, 35, 12)[..., np.newaxis] * 1e-9
tapping_amplitude = 20e-9
harmonic = np.arange(2, 4, 1)

import time

start = time.perf_counter()
alpha = fdm.multilayer.eff_pol_ML(
    z_0, tapping_amplitude, harmonic, beta_stack=beta_stack, t_stack=t_stack
)
end = time.perf_counter()
print(start - end)

plt.plot(z_0, np.abs(alpha))
plt.show()
