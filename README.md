<p align="center">
  <img src="https://github.com/lvgig/tubular/raw/main/logo.png">
</p>

Tubular pre-processing for machine learning!

----

![PyPI](https://img.shields.io/pypi/v/tubular?color=success&style=flat)
![Read the Docs](https://img.shields.io/readthedocs/tubular)
![GitHub](https://img.shields.io/github/license/lvgig/tubular)
![GitHub last commit](https://img.shields.io/github/last-commit/lvgig/tubular)
![GitHub issues](https://img.shields.io/github/issues/lvgig/tubular)
![Build](https://github.com/lvgig/tubular/actions/workflows/python-package.yml/badge.svg?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lvgig/tubular/HEAD?labpath=examples)

`tubular` implements pre-processing steps for tabular data commonly used in machine learning pipelines.

The transformers are compatible with [scikit-learn](https://scikit-learn.org/) [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Each has a `transform` method to apply the pre-processing step to data and a `fit` method to learn the relevant information from the data, if applicable.

The transformers in `tubular` work with data in [pandas](https://pandas.pydata.org/) [DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

There are a variety of transformers to assist with;

- capping
- dates
- imputation
- mapping
- categorical encoding
- numeric operations

Here is a simple example of applying capping to two columns;

```python
from tubular.capping import CappingTransformer
import pandas as pd
from sklearn.datasets import fetch_california_housing

# load the california housing dataset
cali = fetch_california_housing()
X = pd.DataFrame(cali['data'], columns=cali['feature_names'])

# initialise a capping transformer for 2 columns
capper = CappingTransformer(capping_values = {'AveOccup': [0, 10], 'HouseAge': [0, 50]})

# transform the data
X_capped = capper.transform(X)
```

## Installation

The easiest way to get `tubular` is directly from [pypi](https://pypi.org/project/tubular/) with;

 `pip install tubular`

## Documentation

The documentation for `tubular` can be found on [readthedocs](https://tubular.readthedocs.io/en/latest/).

Instructions for building the docs locally can be found in [docs/README](https://github.com/lvgig/tubular/blob/main/docs/README.md).

## Examples

To help get started there are example notebooks in the [examples](https://github.com/lvgig/tubular/tree/main/examples) folder in the repo that show how to use each transformer.

To open the example notebooks in [binder](https://mybinder.org/) click [here](https://mybinder.org/v2/gh/lvgig/tubular/HEAD?labpath=examples) or click on the `launch binder` shield above and then click on the directory button in the side bar to the left to navigate to the specific notebook.

## Issues

For bugs and feature requests please open an [issue](https://github.com/lvgig/tubular/issues).

## Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/). To build the package locally and run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/lvgig/tubular.git
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

See the [CONTRIBUTING](https://github.com/lvgig/tubular/blob/main/CONTRIBUTING.rst) file for the full details of our working practices.
