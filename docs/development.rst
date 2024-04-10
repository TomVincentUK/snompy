.. _development:

Development
===========

Is something missing?
Have you spotted a bug?
Read on for details on how to raise feature requests, or contribute to ``pysnom``.

Feature requests
----------------

Feature requests and bug reports can be submitted by raising an issue on the `pysnom GitHub repository <https://github.com/TomVincentUK/pysnom/issues>`_.

Development process
-------------------

We follow a similar development process to many open-source packages.
New contributors can follow these steps, which are adapted from the `NumPy documentation <https://numpy.org/doc/stable/dev/index.html>`_:

1. Set up a local copy of the repository for development:

   * Go to the `pysnom repository
     <https://github.com/TomVincentUK/pysnom>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone https://github.com/your-username/pysnom.git

   * Change the directory::

      cd pysnom

   * Install pysnom and its dependencies to your Python environment in development mode::

      pip install -e .

   * Add the upstream repository::

      git remote add upstream https://github.com/TomVincentUK/pysnom.git

   * Now, ``git remote -v`` will show two remote repositories named:

     - ``upstream``, which refers to the ``pysnom`` repository
     - ``origin``, which refers to your personal fork

   * Pull the latest changes from upstream, including tags::

      git checkout main
      git pull upstream main --tags

2. Develop your contribution:

   * You might find it helpful to install the dependencies for :ref:`testing<testing>`, :ref:`building documentation<documentation>`, and :ref:`checking your source code<source_code_checks>`.

   * Create a branch for the feature you want to work on.
     Since the branch name will appear in the merge message, use a descriptive name
     such as 'adding-fancy-new-snom-model'::

      git checkout -b 'adding-fancy-new-snom-model'

   * Commit locally as you progress (``git add`` and ``git commit``).
     Use a descriptive commit message, write tests that fail before your change and pass afterward, run all the :ref:`tests locally<testing>`.
     Be sure to document any changed behavior in docstrings.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin adding-fancy-new-snom-model

   * Enter your GitHub username and password (repeat contributors or advanced
     users can remove this step by connecting to GitHub with SSH).

   * Go to GitHub. The new branch will show up with a green Pull Request
     button. Make sure the title and message are clear, concise, and self-
     explanatory. Then click the button to submit it.

4. Review process:

   * The pysnom team will review your pull request.
     If it fits the scope of the project, makes a meaningful contribution, and doesn't break any existing functionality, we will approve it.
     (If you're interested in becoming a team member, feel free to send us a message and we'll be happy to discuss).

   * To update your pull request, make your changes on your local repository, commit, **run tests, and only if they succeed** push to your fork.
     As soon as those changes are pushed up (to the same branch as before) the pull request will update automatically.
     If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a pull request comment.

   * Various continuous integration (CI) services are triggered after each pull request
     update to build the code, run unit tests, measure code coverage and check
     coding style of your branch.
     The CI tests must pass before your pull request can be merged. If CI fails, you can find out why by clicking on the "failed" icon (red cross) and inspecting the build and test log.
     To avoid overuse and waste of this resource, :ref:`test your work<testing>` locally before committing.

   * A pull request must be **approved** by at least one pysnom team member before merging.
     Approval means the team member has carefully reviewed the changes, and the pull request is ready for merging.

.. _testing:

Testing
-------

This project has a suite of tests which run to ensure any changes introduced will not break the intended functionality of the package.
We use the package ``pytest`` to automatically run these tests.
To run the suite you will need to install the testing dependencies with::

   pip install -r requirements_test.txt

You can then run the tests by entering::

   pytest

.. _documentation:

Documentation
-------------

This project uses ``sphinx`` for narrative documentation, and to automatically generate API documentation from docstrings.
To contribute to the documentation you will need to install the necessary dependencies with::

   pip install -r docs/requirements_dev.txt

You can then build the documentation like::

   sphinx-autobuild docs docs/_build/html

Style guide
-----------

.. _source_code_checks:

Source code checks
^^^^^^^^^^^^^^^^^^

This project uses ``black`` to format code, ``isort`` to organise imports, and ``flake8`` for linting.
We also support ``pre-commit`` to ensure these have been run.
To configure your local environment please install these development dependencies and set up the commit hooks like::

   pip install -r requirements_dev.txt
   pre-commit install

Naming conventions
^^^^^^^^^^^^^^^^^^

* Docstrings should follow the `numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

* Function names should describe (in abbreviated English) their return value.
  For example, `eff_pol` instead of `alpha_eff` for "effective polarizability".

* Variable and argument names should match the maths symbol used in the documentation.
  For example, `alpha_eff` instead of `eff_pol` for "effective polarizability", to match the symbol :math:`\alpha_{eff}`.

* Any maths symbols used in the documentation should apply consistently across all functions, and should be added to a "List of symbols" page.
  They should match the symbols used commonly in the literature, except for cases where there are naming conflicts between different authors, and cases where the meaning can be made more clear.
  For example :math:`d_{Q_1'}` instead of :math:`X_1` for the depth of image charge :math:`Q_1'` below the sample.

* Proper nouns should be uncapitalized in variable and function names.
  For example `eff_pol_n_taylor` and `n_lag`, instead of `eff_pol_n_bulk_Taylor` and `n_Lag`, named after Taylor and Laguerre.
  (Classes should be CapWords as per `PEP 8 <https://peps.python.org/pep-0008/#naming-conventions>`_).
