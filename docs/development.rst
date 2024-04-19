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

We follow a similar development process to many open-source packages, allowing contributors to submit changes by raising a pull request on GitHub.

If you're new to this process, or you need a little more guidance, you can follow the steps in this section.

1. Set up a local copy of the repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fork the repository.**
  Go to the `pysnom repository <https://github.com/TomVincentUK/pysnom>`_
  and click the "fork" button to create your own copy of the project.

* **Clone the repository.**
  Open a terminal in the directory where you'd like the project to be stored, then clone the project to your local computer::

    git clone https://github.com/your-username/pysnom.git

* **Link your local copy to the main repository.**
  Change the directory to your newly created local repository::

    cd pysnom

  Now, add the upstream repository::

    git remote add upstream https://github.com/TomVincentUK/pysnom.git

  Then, ``git remote -v`` will show two remote repositories named:

    - ``upstream``, which refers to the ``pysnom`` repository
    - ``origin``, which refers to your personal fork

* **Update your repository.**
  Make sure your local repository is up-to-date, by pulling the latest changes from upstream, including tags::

    git checkout main
    git pull upstream main --tags

2. Configure your Python environment for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Create a fresh environment.**
  We recommend starting with a clean Python environment or virtual environment, to make sure that none of your existing Python packages interfere with ``pysnom``.
  It's common to use tools like `venv <https://docs.python.org/3/library/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ to do this.

* **Install pysnom.**
  In your clean environment, navigate to the pysnom repository and install pysnom and its dependencies to your Python environment::

    pip install -e .

  This command tells `pip` to install ``pysnom`` using the `setup.py` file in the repository's root directory.
  The `-e` flag means it will be installed in development mode, so that changes to the code will show up straight away.

* **Set up formatting tools.**
  We like to keep a consistent and readable coding style, so we use the package ``black`` to format our code, ``isort`` to sort the order of imports, and ``flake8`` to check our code is consistent with the `PEP 8 <https://peps.python.org/pep-0008/>`_ style guide.
  Your code will be automatically rejected unless it's formatted correctly, so we recommend installing these tools on your own computer to check before you submit.
  We also support ``pre-commit`` which automatically checks your code for you when you make a git commit.
  To install these you can type::

    pip install -r requirements_dev.txt
    pre-commit install

  The `-r` flag here tells `pip` to install new packages from a requirements file.
  You might also want to set up your text editor to automatically format your code when you save.

* **Set up testing.**
  On top of format checks, ``pysnom`` also has a suite of tests which run using ``pytest`` to check everything is working.
  Your contributions won't be accepted unless all the tests pass, so we recommend that you setup your environment so you can run the tests on your own computer before you submit your changes::

    pip install -r requirements_test.txt

  Check this has worked before you make any changes by typing::

    pytest

  This should run the tests, which should all pass successfully.
  It will also report the proportion of code lines executed during the tests, which should ideally be 100%.

* **Set up documentation tools.**
  All the features in ``pysnom`` should be documented, so if your edit adds a new feature, or changes how other users will interact with the package, we ask that you also add changes to the documentation to explain it.
  To do this we recommend building the documentation on your own computer.
  We use the package ``sphinx`` to build our documentation.
  You can install this like::

    pip install -r docs/requirements_docs.txt

  You can then build the documentation by typing::

    sphinx-autobuild docs docs/_build/html

  This should build the documentation and host it on a local server, so you can view any changes you make.

3. Develop your contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Create a git branch for the feature you want to work on.**
  Since the branch name will appear in the merge message, use a descriptive name such as 'adding-fancy-new-snom-model'::

    git checkout -b 'adding-fancy-new-snom-model'

* **Commit changes locally as you progress.**
  Use ``git add`` and ``git commit`` with descriptive commit messages.

* **Create tests first.**
  All features in ``pysnom`` should be tested to check they work, so we encourage using `test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_.
  This means the first step when adding a new feature should usually be to create at least one test which will only pass when the new feature works correctly.
  Your new test should be added to the existing test suite in `./pysnom/tests`.
  You can find out how to write good tests by following the `pytest documentation <https://pytest.org/>`_.
  Once your test is created, run the test suite by typing::

    pytest

  This should produce at least one failed test.
  Now you can write your new feature so it passes the test.

  .. hint::

      Some types of changes won't require new tests, such as optimizations of existing routines, or changes that only affect documentation.

* **Add your changes to the package.**
  Make sure that you document your changes by adding or editing docstrings, and if needed, by making changes to the user guide in `./docs/guide`.
  As well as following `PEP 8`_, your code should follow the guidelines in the :ref:`style guide <style_guide>` below.

* **Make your final checks.**
  If your changes are successful, when you run the test suite there should be no errors, and 100% test coverage.
  Also check that all your code is formatted correctly (this should be done automatically when you commit your changes if you installed ``pre-commit`` as above), and that any changes are documented.
  Then you should be ready to submit to the main repository.


4. Submit your contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Push your changes back to your fork on GitHub**::

    git push origin adding-fancy-new-snom-model

  Then enter your GitHub username and password (repeat contributors or advanced users can remove this step by connecting to GitHub with SSH).

* **Make a pull request.**
  Go to GitHub.
  The new branch will show up with a green Pull Request button.
  Make sure the title and message are clear, concise, and self-explanatory.
  Then click the button to submit it.

5. Review process
^^^^^^^^^^^^^^^^^

* **Automatic tests.**
  When you make a pull request, GitHub will run tests to check your code formatting, and it will also run the suite of tests on different operating systems using several versions of Python.

  If a test fails, you can find out why by clicking on the "failed" icon (red cross) and inspecting the build and test log.
  To update your pull request, make your changes on your local repository, commit, then push to your fork.
  As soon as those changes are pushed up (to the same branch as before) the pull request will update automatically.

  The tests must pass for us to merge your changes, so we recommend checking on your own computer before submitting a pull request.

* **The pysnom team will review your pull request**.
  A pull request must be approved by at least one pysnom team member before merging.
  If it fits the scope of the project, makes a meaningful contribution, and doesn't break any existing functionality, we will approve it.
  (If you're interested in becoming a team member, feel free to send us a message and we'll be happy to discuss).

.. _style_guide:

Style guide
-----------

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
  (Classes should be CapWords as per `PEP 8`_).
