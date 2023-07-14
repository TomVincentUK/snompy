pysnom
======
A Python package for modelling contrast in scanning near-field optical microscopy measurements.

Tasks / issues
--------------
* FDM:

  * Bulk:

    * Add a `refl_coef_qs_from_eff_pol` function (just invert it algebraically)

  * Multilayer:

    * Add Lars's version of the effective polarizability (perhaps via a flag in the existing function)

    * Add a Taylor series representation of Lars's version of effective polarizability

    * Add in `refl_coef_qs_from_eff_pol` and `refl_coef_qs` functions

* PDM:

  * Add a Taylor series representation of effective polarizability

  * Add in `refl_coef_qs_from_eff_pol` and `refl_coef_qs` functions

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