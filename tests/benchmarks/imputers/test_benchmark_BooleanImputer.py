"""Speed benchmarks for NumberImputer.

Run with: pytest tests/benchmarks --benchmark-only

These need pytest-benchmark (``benchmark`` dependency group), so the main
test job ignores this directory; they run in the dedicated ``benchmark`` job
in .github/workflows/python-package.yml, which compares a PR against its base.
"""

import pytest

from tests.benchmarks.utils import transform_with_exprs
from tests.utils import dataframe_init_dispatch
from tubular.imputers import BooleanImputer


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_impute_df(request):

    library = request.param

    # test both impute and non impute case
    bool_dict = {"a": [None], "b": [True]}

    return dataframe_init_dispatch(bool_dict, library=library)


def _fitted_transformer(df) -> BooleanImputer:
    return BooleanImputer(impute_value=True, columns=["a", "b"]).fit(df)


def test_transform_single_row(benchmark, get_single_row_impute_df):
    transformer = _fitted_transformer(get_single_row_impute_df)
    transform_exprs = transformer.get_transform_exprs()
    benchmark(transform_with_exprs, transform_exprs, get_single_row_impute_df)
