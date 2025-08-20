Contributing
============

Thanks for your interest in contributing to this package! No contibution is too small! We're hoping it can be made even better through community contributions.

Requests and feedback
---------------------

For any bugs, issues or feature requests please open an `issue <https://github.com/lvgig/tubular/issues>`_ on the project.

Requirements for contributions
------------------------------

We have some general requirements for all contributions then specific requirements when adding completely new transformers to the package. This is to ensure consistency with the existing codebase.

Set up development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For External contributors, first create your own fork of this repo.

Then clone the fork (or this repository if internal);

   .. code::

     git clone https://github.com/lvgig/tubular.git
     cd tubular

Then install tubular and dependencies for development;

   .. code::

     pip install . -r requirements-dev.txt

We use `pre-commit <https://pre-commit.com/>`_ for this project which is configured to check that code is formatted with `black <https://black.readthedocs.io/en/stable/>`_ and passes `ruff <https://beta.ruff.rs/docs/>`_ checks.  For a list of ruff rules follwed by this project check .ruff.toml.

To configure ``pre-commit`` for your local repository run the following;

   .. code::

     pre-commit install

If working in a codespace the dev requirements and precommit will be installed automatically in the dev container.

If you are building the documentation locally you will need the `docs/requirements.txt <https://github.com/lvgig/tubular/blob/main/docs/requirements.txt>`_.

Dependencies
^^^^^^^^^^^^
A point of surprise for some might be that `requirements.txt` and `requirements-dev.txt` are not user-edited files in this repo -
they are compiled using `pip-tools= <https://github.com/jazzband/pip-tools?tab=readme-ov-file#example-usage-for-pip-compile>`_ from
dependencies listed `pyproject.toml`. When adding a new direct dependency, simply add it to the appropriate field inside the package config -
there is no need to pin it, but you can specify a minimum requirement. Then use `pip-compile <https://medium.com/packagr/using-pip-compile-to-manage-dependencies-in-your-python-packages-8451b21a949e>`_
to create a pinned set of dependencies, ensuring reproducibility.

`requirements.txt` and `requirements-dev.txt` are still tracked under source control, despite being 'compiled'.

To compile using `pip-tools`:

  .. code::

     pip install pip-tools # optional
     pip-compile -v --no-emit-index-url --no-emit-trusted-host --output-file requirements.txt  pyproject.toml
     pip-compile --extra dev -v --no-emit-index-url --no-emit-trusted-host --output-file requirements-dev.txt pyproject.toml


General
^^^^^^^

- Please try and keep each pull request to one change or feature only
- Make sure to update the `changelog <https://github.com/lvgig/tubular/blob/main/CHANGELOG.rst>`_ with details of your change

Code formatting
^^^^^^^^^^^^^^^

We use `black <https://black.readthedocs.io/en/stable/>`_ to format our code and follow `pep8 <https://www.python.org/dev/peps/pep-0008/>`_ conventions. 

As mentioned above we use ``pre-commit`` which streamlines checking that code has been formatted correctly.

CI
^^

Make sure that pull requests pass our `CI <https://github.com/lvgig/tubular/actions>`_. It includes checks that;

- code is formatted with `black <https://black.readthedocs.io/en/stable/>`_
- `flake8 <https://flake8.pycqa.org/en/latest/>`_ passes
- the tests for the project pass, with a minimum of 80% branch coverage
- `bandit <https://bandit.readthedocs.io/en/latest/>`_ passes

Tests
^^^^^

We use `pytest <https://docs.pytest.org/en/stable/>`_ as our testing framework.

All existing tests must pass and new functionality must be tested. We aim for 100% coverage on new features that are added to the package.

There are some similarities across the tests for the different transformers in the package. Please refer to existing tests as they give great examples to work from and show what is expected to be covered in the tests.

We also make use of the `test-aide <https://github.com/lvgig/test-aide>`_ package to make mocking easier and to help with generating data when `parametrizing <https://docs.pytest.org/en/6.2.x/parametrize.html>`_ tests for the correct output of transformers' transform methods.

We organise our tests with one script per transformer then group together tests for a particular method into a test class.

Docstrings
^^^^^^^^^^

We follow the `numpy <https://numpydoc.readthedocs.io/en/latest/format.html>`_ docstring style guide.

Docstrings need to be updated for the relevant changes and docstrings need to be added for new transformers.

New transformers
^^^^^^^^^^^^^^^^

Transformers in the package are designed to work with `pandas <https://pandas.pydata.org/>`_ `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ objects.

To be consistent with `scikit-learn <https://scikit-learn.org/stable/data_transforms.html>`_, all transformers must implement at least a  ``transform(X)`` method which applies the data transformation.

If information must be learnt from the data before applying the transform then a ``fit(X, y=None)`` method is required. ``X`` is the input `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ and ``y`` is the response, which may not be required.

Optionally a ``reverse_transform(X)`` method may be appropriate too if there is a way to apply the inverse of the ``transform`` method.

List of contributors
--------------------

For the full list of contributors see the `contributors page <https://github.com/lvgig/tubular/graphs/contributors>`_.

Prior to the open source release of the package there have been contributions from many individuals in the LV GI Data Science team;

- Richard Angell
- Ned Webster
- Dapeng Wang
- David Silverstone
- Shreena Patel
- Angelos Charitidis
- David Hopkinson
- Liam Holmes
- Sandeep Karkhanis
- KarHor Yap
- Alistair Rogers
- Maria Navarro
- Marek Allen
- James Payne
