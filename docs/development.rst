.. _development:

Development
===========

Source code checks
------------------

This project uses ``black`` to format code, ``isort`` to organise imports,
and ``flake8`` for linting.
We also support ``pre-commit`` to ensure these have been run.
To configure your local environment please install these development
dependencies and set up the commit hooks.

.. code-block:: bash

   $ pip install -r requirements_dev.txt
   $ pre-commit install