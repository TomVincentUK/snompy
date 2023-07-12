pysnom
======
A Python package for modelling contrast in scanning near-field optical
microscopy measurements.


Tasks / issues
--------------
* Sample object:
  * Add a way to specify the environment permittivity for beta calculation
  * Add a shorthand way to create a bulk Sample object `.sample.bulk_sample(eps_sub, eps_env=1)`
  * Add tests for reciprocity of single interface eps and beta functions
  * Add far-field Fresnel reflection coefficient calculation
* Tests:
  * Delete obvious tests (e.g. you don't need to test that missing arguments causes an error)
  * Reorganise existing tests to a structure that makes more sense.
* FDM:
  * Reorganise functions
  * Bulk:
    * Add a `refl_coef_qs_from_eff_pol` function (just invert it algebraically)
  * Multilayer:
    * Add Lars's version of the effective polarisability (perhaps via a flag in the existing function)
    * Add a Taylor series representation of Lars's version of effective polarisability
    * Add in `refl_coef_qs_from_eff_pol` and `refl_coef_qs_from_eff_pol_n` functions
* PDM:
  * Add a Taylor series representation of effective polarisability
  * Add in `refl_coef_qs_from_eff_pol` and `refl_coef_qs_from_eff_pol_n` functions
* Main documentation:
  * Finish rules for coding and docs style to development.
  * Add a wishlist to development (e.g. Lightning-rod model).
* Finish API docs:
  * Make sure all public methods and classes have docstrings.
  * Make sure all links work
* Finish/rewrite narrative documentation:
  * Go through all examples and update them to use the new API
  * Go through all symbols and make sure they match API (including in diagrams)
  * Add far-field factor into PMMA example
  * Add captions and alt text to all docs figures
  * Remove pointless PDM/FDM switch
  * Check where you can remove your working (leave details in API rather than guide)