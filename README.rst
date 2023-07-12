pysnom
======
A Python package for modelling contrast in scanning near-field optical
microscopy measurements.


Tasks / issues
--------------
* Finish/rewrite narrative documentation:
  * Add captions and alt text to all docs figures
* Sample object:
  * Add a way to specify the environment permittivity for beta calculation
  * Add a shorthand way to create a bulk Sample object `.sample.bulk_sample(eps_sub, eps_env=1)`
  * Add tests for reciprocity of single interface eps and beta functions
* Tests:
  * Delete obvious tests (e.g. you don't need to test that missing arguments causes an error)
  * Reorganise existing tests to a structure that makes more sense.