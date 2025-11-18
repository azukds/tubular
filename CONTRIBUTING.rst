Contributing
============

Thanks for your interest in contributing to this package! No contibution is too small! We're hoping it can be made even better through community contributions.

Requests and feedback
---------------------

For any bugs, issues or feature requests please open an `issue <https://github.com/azukds/tubular/issues>`_ on the project.

Requirements for contributions
------------------------------

We have some general requirements for all contributions then specific requirements when adding completely new transformers to the package. This is to ensure consistency with the existing codebase.

Set up development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For External contributors, first create your own fork of this repo.

Then clone the fork (or this repository if internal);

   .. code::

     git clone https://github.com/azukds/tubular.git
     cd tubular

Then install tubular and dependencies for development;

   .. code::

     pip install . -r requirements-dev.txt

We use `pre-commit <https://pre-commit.com/>`_ for this project which is configured to check that code is formatted with `black <https://black.readthedocs.io/en/stable/>`_ and passes `ruff <https://beta.ruff.rs/docs/>`_ checks.  For a list of ruff rules follwed by this project check .ruff.toml.

To configure ``pre-commit`` for your local repository run the following;

   .. code::

     pre-commit install

If working in a codespace the dev requirements and precommit will be installed automatically in the dev container.

If you are building the documentation locally you will need the `docs/requirements.txt <https://github.com/azukds/tubular/blob/main/docs/requirements.txt>`_.

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
- Make sure to update the `changelog <https://github.com/azukds/tubular/blob/main/CHANGELOG.rst>`_ with details of your change

Code formatting
^^^^^^^^^^^^^^^

We use `black <https://black.readthedocs.io/en/stable/>`_ to format our code and follow `pep8 <https://www.python.org/dev/peps/pep-0008/>`_ conventions. 

As mentioned above we use ``pre-commit`` which streamlines checking that code has been formatted correctly.

CI
^^

Make sure that pull requests pass our `CI <https://github.com/azukds/tubular/actions>`_. It includes checks that;

- code is formatted with `black <https://black.readthedocs.io/en/stable/>`_
- `flake8 <https://flake8.pycqa.org/en/latest/>`_ passes
- the tests for the project pass, with a minimum of 80% branch coverage
- `bandit <https://bandit.readthedocs.io/en/latest/>`_ passes

Tests
^^^^^

We use `pytest <https://docs.pytest.org/en/stable/>`_ as our testing framework.

We have designed our tests to have a high degree of reusability across classes/usages, but a downside of this is that they can be a bit overwhelming for a newcomer! A few introductory notes on our setup:
- We share tests across classes using fixtures and an inheritance structure
    - tests/conftest.py contains fixtures (uninitialized_transformers, minimal_dataframe_lookup, minimal_attribute_dict) for looking up transformers (and minimal dataframes which they will run on) using the classname
    - We then write parent test classes for shared behaviours, these are written in a generic way so that they just depend on a `transformer_name` attr setup in the child test class.
    - Many of the most universal tests are contained in tests/base_tests.py, so would recommend starting by reading some of the classes in this file and looking at how they are used across the rest of the tests.
- In many cases, we then also reuse tests for different scenarios using pytest.parametrize, these cases include:
    - polars/pandas
    - lazy/eager
    - transformer created from_json/not
- Transformers have class attributes that indicate whether they support the above cases, e.g. `lazyframe_compatible`, and tests for these cases are skipped using tests.utils._check_if_skip_test

As an example, a test class could look like:
```
# parent class avoids pytest 'Tests...' naming, 
# so that test is only run when inherited
class GenericTransformTests:

    # test for both from json/not
    @pytest.mark.parametrize("from_json", [True, False])
    # test for both lazy/not
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    # test for both pandas/polars
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_something(.
      initialized_transformers, # fixture containing transformers to initialize
      minimal_dataframe_lookup, # fixture containing dataframe transformer will run on
      minimal_attribute_dict, # fixture containing minimal init args for transformer
      ):

      args = minimal_attribute_dict[self.transformer_name].copy()

      transformer = uninitialized_transformers[self.transformer_name](**args)

      df = minimal_dataframe_lookup[self.transformer_name]

      # return test as pass if it is not valid to run (e.g. polars test 
      # on non-polars transformer)
      if _check_if_skip_test(x, df, lazy=lazy, from_json=from_json):
            return

      # function handles dumping transformer to json and then
      # loading back before test logic 
      # (if from_json True, otherwise does nothing)
      transformer = _handle_from_json(transformer, from_json)

      # _convert_to_lazy, _collect_frame handle converting to/from
      # lazy before/after test when lazy=True,
      # otherwise they do nothing
      output = transformer.transform(
        _convert_to_lazy(df, lazy)
      )

      output = _collect_frame(df, lazy)

      ... # test something about output

# child class inherits and runs tests, and additional child-specific tests
# can be added here if needed
class TestTransform(GenericTransformTests):

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "FakeTransformer"
```

All existing tests must pass and new functionality must be tested.

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

For the full list of contributors see the `contributors page <https://github.com/azukds/tubular/graphs/contributors>`_.

Prior to the open source release of the package there have been contributions from many individuals in the LV= GI (before becoming part of Allianz Personal) Data Science team:

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
