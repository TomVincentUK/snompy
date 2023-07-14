.. _development:

Development
===========

Naming style
------------

* Function names should describe (in abbreviated English) their return value.
  For example, `eff_pol` instead of `alpha_eff` for "effective polarizability".

* Variable and argument names should match the maths symbol used in the documentation.
  For example, `alpha_eff` instead of `eff_pol` for "effective polarizability", to match the symbol :math:`\alpha_{eff}`.

* Any maths symbols used in the documentation should apply consistently across all functions, and should be added to a "List of symbols" page.
  They should match the symbols used commonly in the literature, except for cases where there are naming conflicts between different authors, and cases where the meaning can be made more clear.
  For example :math:`d_{Q1}` instead of :math:`W_1` for the depth of charge :math:`Q_1` within the tip.

* Proper nouns should be uncapitalised in variable and function names.
  For example `eff_pol_n_taylor` and `n_lag`, instead of `eff_pol_n_bulk_Taylor` and `n_Lag`, named after Taylor and Laguerre.
  (Any classes should be CapWords as per `PEP 8 <https://peps.python.org/pep-0008/#naming-conventions>`_).

**This is not yet implemented, but it should serve as a guide when I rename
my functions and variables.**

Source code checks
------------------

This project uses ``black`` to format code, ``isort`` to organise imports, and ``flake8`` for linting.
We also support ``pre-commit`` to ensure these have been run.
To configure your local environment please install these development dependencies and set up the commit hooks.

.. code-block:: bash

   $ pip install -r requirements_dev.txt
   $ pre-commit install