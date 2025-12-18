import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pandas.api.types import CategoricalDtype
from pandas.testing import assert_series_equal as assert_series_equal_pandas
from polars.testing import assert_series_equal as assert_series_equal_polars

from tubular._utils import new_narwhals_series_with_optimal_pandas_types


class TestNewNarwhalsSeriesWithOptimalPandasTypes:
    @pytest.mark.parametrize(
        ("values", "dtype"),
        [
            ([1, 0, 0], "Int64"),
            ([1, 0, None], "Float64"),
            (["a", "b", None], "String"),
            ((["a", None, "c"], "Categorical")),
            ([True, False, True], "Boolean"),
            ([True, False, None], "Boolean"),
        ],
    )
    def test_polars_unaffected(self, values, dtype):
        "test that polars Series are initialised as usual"
        name = "a"
        output = new_narwhals_series_with_optimal_pandas_types(
            name=name,
            values=values,
            backend="polars",
            dtype=getattr(nw, dtype),
        )

        expected = pl.Series(
            name=name,
            values=values,
            dtype=getattr(pl, dtype),
        )

        assert_series_equal_polars(expected, output.to_native())

    @pytest.mark.parametrize(
        ("values", "polars_dtype", "expected_pandas_dtype"),
        [
            ([1, 0, 0], "Int64", "Int64"),
            ([1, 0, None], "Float64", "Float64"),
            ([True, False, True], "Boolean", "boolean"),
            ([True, False, None], "Boolean", "boolean"),
            (["a", None, "c"], "String", "string"),
            (["a", "b", "c"], "Categorical", "category"),
        ],
    )
    def test_pandas_output(self, values, polars_dtype, expected_pandas_dtype):
        "test that polars Series are initialised as usual"
        name = "a"
        output = new_narwhals_series_with_optimal_pandas_types(
            name=name,
            values=values,
            backend="pandas",
            dtype=getattr(nw, polars_dtype),
        )

        if expected_pandas_dtype == "category":
            expected_pandas_dtype = CategoricalDtype(
                pd.Series(data=values, dtype="string"),
                ordered=False,
            )

        expected = pd.Series(
            name=name,
            data=values,
            dtype=expected_pandas_dtype,
        )

        if expected_pandas_dtype == "string":
            expected = expected.fillna(pd.NA)

        assert_series_equal_pandas(expected, output.to_native())
