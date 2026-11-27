"""Speed benchmarks for ModeImputer.

Run with: pytest tests/benchmarks --benchmark-only

These need pytest-benchmark (``benchmark`` dependency group), so the main
test job ignores this directory; they run in the dedicated ``benchmark`` job
in .github/workflows/python-package.yml, which compares a PR against its base.
"""

import narwhals as nw
import pytest

from tests.benchmarks.utils import transform_with_exprs
from tests.utils import dataframe_init_dispatch
from tubular.imputers import ModeImputer


@pytest.fixture(scope="module", params=["pandas", "polars"])
def get_single_row_impute_df(request):
    "Setup single row dataframe covering main cases for transform."

    library = request.param

    # test both impute and non impute case
    numeric_dict = {"a": [None], "b": [1]}

    df = dataframe_init_dispatch(numeric_dict, library=library)

    df = nw.from_native(df)

    return df.with_columns(nw.col("a").cast(nw.Float64), nw.col("b").cast(nw.Float64))


def _fitted_transformer() -> ModeImputer:
    """Set up transformer instance ready to transform.

    For ease/speed in pipeline, avoid fitting and directly
    set any needed attrs.
    """
    transformer = ModeImputer(columns=["a", "b"])
    transformer.impute_values_ = {"a": 1, "b": 2}
    return transformer


def test_transform_single_row(benchmark, get_single_row_impute_df):
    """Benchmark single row transform.

    (using exprs, as this is recommended fast approach).
    """
    transformer = _fitted_transformer()
    transform_exprs = transformer.get_transform_exprs()
    benchmark(transform_with_exprs, transform_exprs, get_single_row_impute_df)
