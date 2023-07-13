pysnom
======
A Python package for modelling contrast in scanning near-field optical
microscopy measurements.


Tasks / issues
--------------
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
  * Add captions and alt text to all docs figures
  * Remove pointless PDM/FDM switch
  * Check where you can remove your working (leave details in API rather than guide)