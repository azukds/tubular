"Fixtures for benchmark tests."

import pytest

from tests.benchmarks.benchmark_datasets import (
    _get_boolean_dict,
    _get_numeric_dict,
    _get_string_dict,
)
from tests.utils import dataframe_init_dispatch


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_numeric_df(request):

    library = request.param

    numeric_dict = _get_numeric_dict(1)

    return dataframe_init_dispatch(numeric_dict, library=library)


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_string_df(request):

    library = request.param

    string_dict = _get_string_dict(1)

    return dataframe_init_dispatch(string_dict, library=library)


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_boolean_df(request):

    library = request.param

    boolean_dict = _get_boolean_dict(1)

    return dataframe_init_dispatch(boolean_dict, library=library)
