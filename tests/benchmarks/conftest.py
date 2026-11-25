"Fixtures for benchmark tests."

import pytest

from tests.benchmarks.benchmark_datasets import _get_numeric_dict
from tests.utils import dataframe_init_dispatch


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_large_numeric_df(request):

    library = request.param

    numeric_dict = _get_numeric_dict(100000)

    return dataframe_init_dispatch(numeric_dict, library=library)


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_numeric_df(request):

    library = request.param

    numeric_dict = _get_numeric_dict(1)

    return dataframe_init_dispatch(numeric_dict, library=library)
