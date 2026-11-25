"""Speed benchmarks for NumberImputer.

Run with: pytest tests/benchmarks --benchmark-only

These need pytest-benchmark (``benchmark`` dependency group), so the main
test job ignores this directory; they run in the dedicated ``benchmark`` job
in .github/workflows/python-package.yml, which compares a PR against its base.
"""

from tests.benchmarks.utils import transform_with_exprs
from tubular.imputers import NumberImputer


def _fitted_transformer(df) -> NumberImputer:
    return NumberImputer(impute_value=-1, columns=["a", "b", "c"]).fit(df)


def test_transform_single_row(benchmark, get_single_row_numeric_df):
    transformer = _fitted_transformer(get_single_row_numeric_df)
    transform_exprs = transformer.get_transform_exprs()
    benchmark(transform_with_exprs, transform_exprs, get_single_row_numeric_df)
