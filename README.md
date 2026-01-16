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

transformer = CappingTransformer(
    capping_values={"a": [10, 20], "b": [1, 3]},
)

test_df = pl.DataFrame({"a": [1, 15, 18, 25], "b": [6, 2, 7, 1], "c": [1, 2, 3, 4]})

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
Tubular also supports saving/reading transformers and pipelines to/from json format (goodbye .pkls!), which we demo below:

```python
import polars as pl
from tubular.imputers import MeanImputer, MedianImputer
from sklearn.pipeline import Pipeline
from tubular.pipeline import dump_pipeline_to_json, load_pipeline_from_json

# Create a simple dataframe

df = pl.DataFrame({"a": [1, 5], "b": [10, None]})

# Add imputers
median_imputer = MedianImputer(columns=["b"])
mean_imputer = MeanImputer(columns=["b"])

# Create and fit the pipeline
original_pipeline = Pipeline(
    [("MedianImputer", median_imputer), ("MeanImputer", mean_imputer)]
)
original_pipeline = original_pipeline.fit(df)

# Dumping the pipeline to JSON
pipeline_json = dump_pipeline_to_json(original_pipeline)
pipeline_json

# Printed value:
# ->
# {
# 'MedianImputer': {
#     'tubular_version': '2.6.1',
#     'classname': 'MedianImputer',
#     'init': {
#          'columns': ['b'],
#          'copy': False,
#          'verbose': False,
#          'return_native': True,
#          'weights_column': None
#          },
#     'fit': {
#           'impute_values_': {'b': 10.0}
#           }
#      },
# 'MeanImputer': {
#      'tubular_version': '2.6.1',
#      'classname': 'MeanImputer',
#      'init': {
#          'columns': ['b'],
#          'copy': False,
#          'verbose': False,
#          'return_native': True,
#          'weights_column': None
#           },
#      'fit': {
#          'impute_values_': {
#          'b': 10.0
#          }
#     }
# }

# Load the pipeline from JSON
pipeline = load_pipeline_from_json(pipeline_json)

# Verify the reconstructed pipeline
print(pipeline)

# Printed value:
# Pipeline(steps=[('MedianImputer', MedianImputer(columns=['b'])),
#                 ('MeanImputer', MeanImputer(columns=['b']))])
```

We are currently in the process of rolling out support for polars lazyframes!

track our progress below:

<!-- AUTO-GENERATED feature table -->
|                                    | polars_compatible   | pandas_compatible   | jsonable           | lazyframe_compatible   |
|------------------------------------|---------------------|---------------------|--------------------|------------------------|
| AggregateColumnsOverRowTransformer | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| AggregateRowsOverColumnTransformer | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| ArbitraryImputer                   | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| BetweenDatesTransformer            | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| CappingTransformer                 | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| CompareTwoColumnsTransformer       | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| DateDifferenceTransformer          | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| DatetimeComponentExtractor         | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| DatetimeInfoExtractor              | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| DatetimeSinusoidCalculator         | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| DifferenceTransformer              | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| GroupRareLevelsTransformer         | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| MappingTransformer                 | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| MeanImputer                        | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| MeanResponseTransformer            | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| MedianImputer                      | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| ModeImputer                        | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| NullIndicator                      | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| OneDKmeansTransformer              | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| OneHotEncodingTransformer          | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| OutOfRangeNullTransformer          | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :x:                    |
| RatioTransformer                   | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| SetValueTransformer                | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| ToDatetimeTransformer              | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
| WhenThenOtherwiseTransformer       | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:     |
<!-- /AUTO-GENERATED feature table -->

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
