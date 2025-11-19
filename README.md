<p align="center">
  <img src="https://github.com/azukds/tubular/raw/main/logo.png">
</p>

Feature engineering on polars and pandas dataframes for machine learning!

----

![PyPI](https://img.shields.io/pypi/v/tubular?color=success&style=flat)
![Read the Docs](https://img.shields.io/readthedocs/tubular)
![GitHub](https://img.shields.io/github/license/azukds/tubular)
![GitHub last commit](https://img.shields.io/github/last-commit/azukds/tubular)
![GitHub issues](https://img.shields.io/github/issues/azukds/tubular)
![Build](https://github.com/azukds/tubular/actions/workflows/python-package.yml/badge.svg?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/azukds/tubular/HEAD?labpath=examples)

`tubular` implements pre-processing steps for tabular data commonly used in machine learning pipelines.

The transformers are compatible with [scikit-learn](https://scikit-learn.org/) [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Each has a `transform` method to apply the pre-processing step to data and a `fit` method to learn the relevant information from the data, if applicable.

The transformers in `tubular` are written in narwhals [narwhals](https://narwhals-dev.github.io/narwhals/), so are agnostic between [pandas](https://pandas.pydata.org/) and [polars](https://pola.rs/) dataframes, and will utilise the chosen (pandas/polars) API under the hood.

There are a variety of transformers to assist with;

- capping
- dates
- imputation
- mapping
- categorical encoding
- numeric operations

Here is a simple example of applying capping to two columns;

```python
import polars as pl

transformer=CappingTransformer(
capping_values={'a': [10, 20], 'b': [1,3]},
  )

test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

transformer.transform(test_df)
# ->
# shape: (4, 3)
# ┌─────┬─────┬─────┐
# │ a   ┆ b   ┆ c   │
# │ --- ┆ --- ┆ --- │
# │ i64 ┆ i64 ┆ i64 │
# ╞═════╪═════╪═════╡
# │ 10  ┆ 3   ┆ 1   │
# │ 15  ┆ 2   ┆ 2   │
# │ 18  ┆ 3   ┆ 3   │
# │ 20  ┆ 1   ┆ 4   │
# └─────┴─────┴─────┘
```

## Installation

The easiest way to get `tubular` is directly from [pypi](https://pypi.org/project/tubular/) with;

 `pip install tubular`

## Documentation

The documentation for `tubular` can be found on [readthedocs](https://tubular.readthedocs.io/en/latest/).

Instructions for building the docs locally can be found in [docs/README](https://github.com/azukds/tubular/blob/main/docs/README.md).

## Examples

We utilise [doctest](https://docs.python.org/3/library/doctest.html) to keep valid usage examples in the docstrings of transformers in the package, so please see these for getting started!

## Issues

For bugs and feature requests please open an [issue](https://github.com/azukds/tubular/issues).

## Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/). To build the package locally and run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/azukds/tubular.git
cd tubular
```

Next install `tubular` and development dependencies;

```shell
pip install . -r requirements-dev.txt
```

Finally run the test suite with `pytest`;

```shell
pytest
```

## Contribute

`tubular` is under active development, we're super excited if you're interested in contributing! 

See the [CONTRIBUTING](https://github.com/azukds/tubular/blob/main/CONTRIBUTING.rst) file for the full details of our working practices.
