import datetime
import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
    create_date_diff_different_dtypes,
)
from tests.utils import assert_frame_equal_dispatch
from tubular.dates import BetweenDatesTransformer


class TestInit(
    ColumnStrListInitTests,
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
):
    "tests for BetweenDatesTransformer.__init__."

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("upper_inclusive", 1000),
            ("lower_inclusive", "hi"),
        ],
    )
    def test_inclusive_args_non_bool_error(self, param, value):
        """Test that an exception is raised if upper_inclusive not a bool."""

        param_dict = {param: value}
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            BetweenDatesTransformer(
                columns=["a", "b", "c"],
                new_column_name="d",
                **param_dict,
            )

    @pytest.mark.parametrize(
        "columns",
        [
            ["a", "b"],
            ["a", "b", "c", "d"],
        ],
    )
    def test_wrong_col_count_error(self, columns):
        """Test that an exception is raised if too many/too few columns."""

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            BetweenDatesTransformer(
                columns=columns,
                new_column_name="d",
            )


def expected_df_1(library="pandas"):
    """Expected output from transform in test_output."""
    df = d.create_is_between_dates_df_1(library=library)

    df = nw.from_native(df)
    native_backend = nw.get_native_namespace(df)
    df = df.with_columns(
        nw.new_series(
            name="d",
            values=[True, False],
            backend=native_backend.__name__,
        ),
    )

    return df.to_native()


def expected_df_2(library="pandas"):
    """Expected output from transform in test_output_both_exclusive."""
    df = d.create_is_between_dates_df_2(library=library)

    df = nw.from_native(df)
    native_backend = nw.get_native_namespace(df)
    df = df.with_columns(
        nw.new_series(
            name="e",
            values=[False, False, True, True, False, False],
            backend=native_backend.__name__,
        ),
    )

    return df.to_native()


def expected_df_3(library="pandas"):
    """Expected output from transform in test_output_lower_exclusive."""
    df = d.create_is_between_dates_df_2(library=library)

    df = nw.from_native(df)
    native_backend = nw.get_native_namespace(df)
    df = df.with_columns(
        nw.new_series(
            name="e",
            values=[False, False, True, True, True, False],
            backend=native_backend.__name__,
        ),
    )

    return df.to_native()


def expected_df_4(library="pandas"):
    """Expected output from transform in test_output_upper_exclusive."""
    df = d.create_is_between_dates_df_2(library=library)

    df = nw.from_native(df)
    native_backend = nw.get_native_namespace(df)
    df = df.with_columns(
        nw.new_series(
            name="e",
            values=[False, True, True, True, False, False],
            backend=native_backend.__name__,
        ),
    )

    return df.to_native()


def expected_df_5(library="pandas"):
    """Expected output from transform in test_output_both_inclusive."""
    df = d.create_is_between_dates_df_2(library=library)

    df = nw.from_native(df)
    native_backend = nw.get_native_namespace(df)
    df = df.with_columns(
        nw.new_series(
            name="e",
            values=[False, True, True, True, True, False],
            backend=native_backend.__name__,
        ),
    )

    return df.to_native()


class TestTransform(
    GenericTransformTests,
    GenericDatesMixinTransformTests,
    DropOriginalTransformMixinTests,
):
    """Tests for BetweenDatesTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_is_between_dates_df_1(library="pandas"),
                expected_df_1(library="pandas"),
            ),
            (
                d.create_is_between_dates_df_1(library="polars"),
                expected_df_1(library="polars"),
            ),
        ],
    )
    def test_output(self, df, expected):
        """Test the output of transform is as expected."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="d",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_is_between_dates_df_2(library="pandas"),
                expected_df_2(library="pandas"),
            ),
            (
                d.create_is_between_dates_df_2(library="polars"),
                expected_df_2(library="polars"),
            ),
        ],
    )
    def test_output_both_exclusive(self, df, expected):
        """Test the output of transform is as expected if both limits are exclusive."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_is_between_dates_df_2(library="pandas"),
                expected_df_3(library="pandas"),
            ),
            (
                d.create_is_between_dates_df_2(library="polars"),
                expected_df_3(library="polars"),
            ),
        ],
    )
    def test_output_lower_exclusive(self, df, expected):
        """Test the output of transform is as expected if the lower limits are exclusive only."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=True,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_is_between_dates_df_2(library="pandas"),
                expected_df_4(library="pandas"),
            ),
            (
                d.create_is_between_dates_df_2(library="polars"),
                expected_df_4(library="polars"),
            ),
        ],
    )
    def test_output_upper_exclusive(self, df, expected):
        """Test the output of transform is as expected if the upper limits are exclusive only."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_is_between_dates_df_2(library="pandas"),
                expected_df_5(library="pandas"),
            ),
            (
                d.create_is_between_dates_df_2(library="polars"),
                expected_df_5(library="polars"),
            ),
        ],
    )
    def test_output_both_inclusive(self, df, expected):
        """Test the output of transform is as expected if the both limits are inclusive."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=True,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    def test_warning_message(self):
        """Test a warning is generated if not all the values in column_upper are greater than or equal to column_lower."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=True,
        )

        df = d.create_is_between_dates_df_2()
        df = nw.from_native(df)

        df = (
            df.with_row_index("i")
            .with_columns(
                c=nw.when(nw.col("i") == 0)
                .then(datetime.datetime(1989, 3, 1, tzinfo=datetime.timezone.utc))
                .otherwise("c"),
            )
            .drop("i")
        ).to_native()

        with pytest.warns(Warning, match="not all c are greater than or equal to a"):
            x.transform(df)

    @pytest.mark.parametrize(
        ("columns"),
        [
            ["a_date", "b_date", "c_date"],
            ["a_date", "b_date", "c_datetime"],
            ["a_date", "b_datetime", "c_datetime"],
            ["a_datetime", "b_date", "c_date"],
            ["a_datetime", "b_date", "c_datetime"],
            ["a_datetime", "b_datetime", "c_date"],
        ],
    )
    @pytest.mark.parametrize(
        ("library"),
        ["pandas", "polars"],
    )
    def test_output_different_date_dtypes(self, columns, library):
        """Test the output of transform is as expected if both limits are exclusive."""
        x = BetweenDatesTransformer(
            columns=columns,
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df = d.create_is_between_dates_df_3(library=library)
        output = [False, False, True, True, False, False]
        df = nw.from_native(df)
        expected = df.clone()
        if library == "pandas":
            expected = expected.with_columns(
                [
                    nw.from_native(pd.Series(output), series_only=True).alias("e"),
                ],
            ).to_native()
        if library == "polars":
            expected = expected.with_columns(
                [
                    nw.from_native(pl.Series(output), series_only=True).alias("e"),
                ],
            ).to_native()

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    # overloading below test as column count is different for this one
    @pytest.mark.parametrize(
        ("columns, datetime_col, date_col"),
        [
            (["date_col_1", "datetime_col_2", "date_col_2"], 1, 0),
            (["datetime_col_1", "date_col_2", "datetime_col_2"], 0, 1),
        ],
    )
    @pytest.mark.parametrize(
        ("library"),
        ["pandas", "polars"],
    )
    def test_mismatched_datetypes_error(
        self,
        columns,
        datetime_col,
        date_col,
        uninitialized_transformers,
        library,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"

        x = uninitialized_transformers[self.transformer_name](
            columns=columns,
            new_column_name="c",
        )

        df = create_date_diff_different_dtypes(library=library)

        present_types = (
            {"datetime64", "date32[pyarrow]"}
            if datetime_col == 0
            else {"date32[pyarrow]", "datetime64"}
        )
        msg = re.escape(
            f"Columns fed to datetime transformers should be ['datetime64', 'date32[pyarrow]'] and have consistent types, but found {present_types}. Please use ToDatetimeTransformer to standardise",
        )
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            x.transform(df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"
