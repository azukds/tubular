import pandas as pd
import pytest

from tubular._utils import _assess_pandas_object_column


class TestAssessPandasObjectColumn:
    "tests for _assess_pandas_object_column function"

    def test_boolean_case(self):
        "test function output for bool/null column"

        column = "a"

        df = pd.DataFrame({column: [True, False, None]})

        expected_pandas_col_type, expected_polars_col_type = "bool", "Boolean"

        actual_pandas_col_type, actual_polars_col_type = _assess_pandas_object_column(
            df,
            column,
        )

        assert expected_pandas_col_type == actual_pandas_col_type, (
            f"_assess_pandas_object_column output not as expected for pandas bool type, expected {expected_pandas_col_type} but got {actual_pandas_col_type}"
        )

        assert expected_polars_col_type == actual_polars_col_type, (
            f"_assess_pandas_object_column output not as expected for polars bool type, expected {expected_polars_col_type} but got {actual_polars_col_type}"
        )

    def test_null_case(self):
        "test function output for null column"

        column = "a"

        df = pd.DataFrame({column: [None, None]})

        expected_pandas_col_type, expected_polars_col_type = "null", "Unknown"

        actual_pandas_col_type, actual_polars_col_type = _assess_pandas_object_column(
            df,
            column,
        )

        assert expected_pandas_col_type == actual_pandas_col_type, (
            f"_assess_pandas_object_column output not as expected for pandas null type, expected {expected_pandas_col_type} but got {actual_pandas_col_type}"
        )

        assert expected_polars_col_type == actual_polars_col_type, (
            f"_assess_pandas_object_column output not as expected for polars null type, expected {expected_polars_col_type} but got {actual_polars_col_type}"
        )

    def test_object_case(self):
        "test function output for object column (not one of our more specific subcases)"

        column = "a"

        df = pd.DataFrame({column: [["a"], ["b"]]})

        expected_pandas_col_type, expected_polars_col_type = "object", "Object"

        actual_pandas_col_type, actual_polars_col_type = _assess_pandas_object_column(
            df,
            column,
        )

        assert expected_pandas_col_type == actual_pandas_col_type, (
            f"_assess_pandas_object_column output not as expected for pandas null type, expected {expected_pandas_col_type} but got {actual_pandas_col_type}"
        )

        assert expected_polars_col_type == actual_polars_col_type, (
            f"_assess_pandas_object_column output not as expected for polars null type, expected {expected_polars_col_type} but got {actual_polars_col_type}"
        )

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2],
            [True, False],
        ],
    )
    def test_errors_for_non_object(self, values):
        "test function errors for non object columns"

        column = "a"

        df = pd.DataFrame({column: values})

        msg = "_assess_pandas_object_column only works with object dtype columns"

        with pytest.raises(
            TypeError,
            match=msg,
        ):
            _, _ = _assess_pandas_object_column(df, column)
