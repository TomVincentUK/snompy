finite-dipole
=============
Finite dipole model for scanning near-field optical microscopy contrast


Tasks / issues
--------------
-  Write narrative documentation
-  Add examples to `bulk` and `multilayer` docstrings
-  Add Notes section to `demod` to replace horrid pseudocode explanation
-  Add See also sections to `reflection` functions


Developing
----------

This project uses ``black`` to format code and ``flake8`` for linting. We
also support ``pre-commit`` to ensure these have been run. To configure
your local environment please install these development dependencies and
set up the commit hooks.

.. code-block:: bash

   $ pip install black flake8 pre-commit
   $ pre-commit install