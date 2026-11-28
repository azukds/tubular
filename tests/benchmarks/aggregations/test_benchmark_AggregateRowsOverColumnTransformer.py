"""Speed benchmarks for AggregateRowsOverColumnTransformer.

Run with: pytest tests/benchmarks --benchmark-only

These need pytest-benchmark (``benchmark`` dependency group), so the main
test job ignores this directory; they run in the dedicated ``benchmark`` job
in .github/workflows/python-package.yml, which compares a PR against its base.
"""

import narwhals as nw
import pytest

from tests.benchmarks.utils import transform_with_exprs
from tests.utils import dataframe_init_dispatch
from tubular.aggregations import AggregateRowsOverColumnTransformer


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_aggregation_df(request):
    "Setup single row dataframe covering main cases for transform."

    library = request.param

    # one null and one non null column to aggregate, grouped by key
    numeric_dict = {"a": [None], "b": [1], "key": [1]}

    df = dataframe_init_dispatch(numeric_dict, library=library)

    df = nw.from_native(df)

    return df.with_columns(
        nw.col("a").cast(nw.Float64),
        nw.col("b").cast(nw.Float64),
        nw.col("key").cast(nw.Int64),
    )


def _fitted_transformer() -> AggregateRowsOverColumnTransformer:
    """Set up transformer instance ready to transform.

    This transformer does not fit, so it is ready after init.
    Uses every supported aggregation.
    """
    return AggregateRowsOverColumnTransformer(
        columns=["a", "b"],
        aggregations=["min", "max", "mean", "sum", "median", "count"],
        key="key",
    )


def test_transform_single_row(benchmark, get_single_row_aggregation_df):
    """Benchmark single row transform.

    (using exprs, as this is recommended fast approach).
    """
    transformer = _fitted_transformer()
    transform_exprs = transformer.get_transform_exprs()
    benchmark(transform_with_exprs, transform_exprs, get_single_row_aggregation_df)
