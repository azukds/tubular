import datetime

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.dates import DateDiffLeapYearTransformer


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    TwoColumnListInitTests,
):
    """Tests for DateDiffLeapYearTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"

    def test_missing_replacement_type_error(self):
        """Test that an exception is raised if missing_replacement is not the correct type."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DateDiffLeapYearTransformer(
                columns=["dummy_1", "dummy_2"],
                new_column_name="dummy_3",
                drop_original=False,
                missing_replacement=[1, 2, 3],
            )


def expected_df_1(library="pandas"):
    """Expected output for test_expected_output_drop_original_true."""

    df_dict = {
        "c": [
            26,
            19,
            0,
            0,
            0,
            -2,
            -3,
            30,
        ],
    }

    if library == "pandas":
        return pd.DataFrame(df_dict, dtype="int64[pyarrow]")

    return dataframe_init_dispatch(dataframe_dict=df_dict, library=library)


def expected_df_2(library="pandas"):
    """Expected output for test_expected_output_drop_original_false."""

    df_dict = {
        "a": [
            datetime.date(1993, 9, 27),  # day/month greater than
            datetime.date(2000, 3, 19),  # day/month less than
            datetime.date(2018, 11, 10),  # same day
            datetime.date(2018, 10, 10),  # same year day/month greater than
            datetime.date(2018, 10, 10),  # same year day/month less than
            datetime.date(2018, 10, 10),  # negative day/month less than
            datetime.date(2018, 12, 10),  # negative day/month greater than
            datetime.date(
                1985,
                7,
                23,
            ),  # large gap, this is incorrect with timedelta64 solutions
        ],
        "b": [
            datetime.date(2020, 5, 1),
            datetime.date(2019, 12, 25),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 9, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 7, 23),
        ],
        "c": [
            26,
            19,
            0,
            0,
            0,
            -2,
            -3,
            30,
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    # ensure types line up with test data
    df = nw.from_native(df)
    for col in [col for col in df.columns if col not in ["c"]]:
        df = df.with_columns(
            nw.col(col).cast(nw.Date),
        )

    if library == "pandas":
        df = nw.to_native(df)
        df["c"] = df["c"].astype("int64[pyarrow]")
        return df

    return nw.to_native(df)


# add the expected to fix float to int with results
def expected_date_diff_df_2(library="pandas"):
    """Expected output for test_expected_output_nans_in_data."""

    df_dict = {
        "c": [
            pd.NA if library == "pandas" else None,
            19,
            0,
            0,
            0,
            -2,
            -3,
            30,
        ],
    }

    if library == "pandas":
        return pd.DataFrame(df_dict, dtype="int64[pyarrow]")

    return pl.DataFrame(df_dict)


# add the expected to fix float to int with results
def expected_date_diff_df_3(library="pandas"):
    """Expected output for test_expected_output_nans_in_data with missing replace with 0."""

    df_dict = {
        "c": [
            0,
            19,
            0,
            0,
            0,
            -2,
            -3,
            30,
        ],
    }

    if library == "pandas":
        return pd.DataFrame(df_dict, dtype="int64[pyarrow]")

    return pl.DataFrame(df_dict)


class TestTransform(
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    GenericDatesMixinTransformTests,
):
    """Tests for DateDiffLeapYearTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (d.create_date_test_df(library="pandas"), expected_df_1(library="pandas")),
            (d.create_date_test_df(library="polars"), expected_df_1(library="polars")),
        ],
    )
    def test_expected_output_drop_original_true(self, df, expected):
        """Test that the output is expected from transform, when drop_original is True.

        This tests positive year gaps, negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=True,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (d.create_date_test_df(library="pandas"), expected_df_2(library="pandas")),
            (d.create_date_test_df(library="polars"), expected_df_2(library="polars")),
        ],
    )
    def test_expected_output_drop_original_false(self, df, expected):
        """Test that the output is expected from transform, when drop_original is False.

        This tests positive year gaps , negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("columns"),
        [
            ["date_col_1", "date_col_2"],
            ["datetime_col_1", "datetime_col_2"],
        ],
    )
    def test_expected_output_nans_in_data(self, columns, library):
        "Test that transform works for different date datatype combinations with nans in data"
        x = DateDiffLeapYearTransformer(
            columns=columns,
            new_column_name="c",
            drop_original=True,
        )

        expected = expected_date_diff_df_2(library=library)

        df = d.create_date_diff_different_dtypes_and_nans(library=library)

        df_transformed = x.transform(df[columns])

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_nans_in_data_with_replace(self, library):
        "Test that transform works for different date datatype combinations with nans in data and replace nans"
        x = DateDiffLeapYearTransformer(
            columns=["date_col_1", "date_col_2"],
            new_column_name="c",
            drop_original=True,
            missing_replacement=0,
        )

        expected = expected_date_diff_df_3(library=library)

        df = d.create_date_diff_different_dtypes_and_nans(library=library)

        df_transformed = x.transform(df[["date_col_1", "date_col_2"]])

        assert_frame_equal_dispatch(df_transformed, expected)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"
