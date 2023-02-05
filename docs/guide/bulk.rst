Bulk finite dipole model
========================

* SNOM tip and sample in laser -> sigma = (1 + c*r)^2 eff_pol
* We want to find eff_pol, here: `finite-dipole.bulk.eff_pol_0`
* To remove the far-field, we need to tap and demodulate so we want to find eff_pol_n, here: `finite-dipole.bulk.eff_pol`

Principles of the finite dipole model
-------------------------------------

* AFM tip in laser -> ellipsoid in E-field
* Quasi-static approximation (E-fields change slow enough that the system is always in equilibrium)
* E-field around ellipsoid looks like finite dipole (to contrast with point dipole)
* Only charge at end is close to sample, so can be treated as a point charge
* Charge in end of tip induces an image charge, which induces another charge in the tip
* That charge also has it's own image charge and counter-charge (ref Cvitkovic, all above)
* The system can be solved for the effective polarisability as: eqn (ref Hauer)
* geom function: eqn (ref Hauer)
* Properties of eff_pol_0:
  * Complex number -> amplitude and phase
  * Decays non-linearly from sample surface
  * Depends on dielectric functions of sample and environment


Removing far-field signals by demodulation
------------------------------------------

* Why we need to demodulate (ref Keilmann)
* Higher harmonics lead to greater near-field confinement (ref Keilmann)
* This can be exploited for depth-sensitive probing of materials (ref Lars)
* Warn that there's still a far-field contribution (ref other Lars)
* For more details on how to demodulate see specific section on guide.

References
----------
Keilmann
Lars depth sensitivity
Lars ratios of harmonics