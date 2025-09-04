"""This module contains functions that create simple datasets that are used in the tests."""

import datetime

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl

import tests.utils as u


def create_numeric_df_1(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [34.48, 21.71, 32.83, 1.08, 32.93, 4.74, 2.76, 75.7, 14.08, 61.31],
        "b": [12.03, 20.32, 24.12, 24.18, 68.99, 0.0, 0.0, 59.46, 11.02, 60.68],
        "c": [17.06, 12.25, 19.15, 29.73, 1.98, 8.23, 15.22, 20.59, 3.82, 39.73],
        "d": [25.94, 70.22, 72.94, 64.55, 0.41, 13.62, 30.22, 4.6, 67.13, 10.38],
        "e": [94.3, 4.18, 51.7, 16.63, 2.6, 16.57, 3.51, 30.79, 66.19, 25.44],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_numeric_df_2(library="pandas"):
    """Example with numeric dataframe that includes missings."""

    df_dict = {
        "a": [2, 3, 2, 1, 4, 1],
        "b": [None, None, 1, 3, 3, 4],
        "c": [1, 1, 2, 3, 3, None],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_object_df(library="pandas"):
    """Example with object columns - c is numeric target"""

    df_dict = {
        "a": [1, 2, 3, 4, 5],
        "b": ["f", "g", "h", "i", "j"],
        "c": ["a", "b", "c", "d", "e"],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_df_1(library="pandas"):
    """Create simple DataFrame with the following...

    6 rows
    2 columns;
    - a integer 1:6
    - b object a:f
    no nulls
    """

    df_dict = {"a": [1, 2, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]}

    return u.dataframe_init_dispatch(df_dict, library)


def create_df_2(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3, 4, 5, 6, None],
        "b": ["a", "b", "c", "d", "e", "f", None],
        "c": ["a", "b", "c", "d", "e", "f", None],
    }

    df = u.dataframe_init_dispatch(df_dict, library)
    if library == "pandas":
        df["c"] = df["c"].astype("category")
    elif library == "polars":
        df = df.with_columns(df["c"].cast(pl.Categorical))
    return df


def create_df_3(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3, 4, 5, 6, None],
        "b": [1, 2, 3, None, 7, 8, 9],
        "c": [None, 1, 2, 3, -4, -5, -6],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_df_4(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3, 4, 5, 6, 7, None],
        "b": ["a", "b", "c", "d", "e", "f", "g", None],
        "c": ["a", "b", "c", "d", "e", "f", "g", None],
    }

    df = u.dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_df_5(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None],
        "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", None],
    }

    df = u.dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_df_6(library="pandas"):
    """Nulls in different positions to check summing weights by col with nulls."""

    df_dict = {
        "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None],
        "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", None],
    }

    df = u.dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_df_7(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [4, 2, 2, 1, 3],
        "b": ["x", "z", "y", "x", "x"],
        "c": ["c", "a", "a", "c", "b"],
    }

    df = u.dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_df_8(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 5, 2, 3, 3],
        "b": ["w", "w", "z", "y", "x"],
        "c": ["a", "a", "c", "b", "a"],
    }

    df = u.dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)

    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_df_9(library="pandas"):
    """Create simple DataFrame to use in other tests."""
    df_dict = {
        "a": [1, 2, None, 4, None, 6],
        "b": [None, 5, 4, 3, 2, 1],
        "c": [3, 2, 1, 4, 5, 6],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_df_10(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None],
        "c": [1, 1, 3, 4, 5, 6, 5, 8, 9, 50],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_df_11(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_bool_and_float_df(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [1.0, 2.0, None],
        "b": [True, False, None],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_df_with_none_and_nan_cols(library="pandas"):
    """Create simple DataFrame to use in other tests."""

    df_dict = {
        "a": [np.nan, np.nan, np.nan],
        "b": [None, None, None],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_aggregate_over_rows_test_df(library="pandas"):
    "create a minimal test df for AggregateRowsOverColumnTransformer"

    df_dict = {
        "a": [1, 2, 3, 4, 8],
        "b": [2, 3, 4, 5, 9],
        "c": ["A", "B", "A", "B", "A"],
    }

    return u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)


def create_weighted_imputers_test_df(library="pandas"):
    """Create DataFrame to use imputer tests that correct values are imputed for weighted dataframes.

    weight column contains the weights between 0 and 1
    """

    df_dict = {
        "a": [1.0, 1.0, 1.0, 3.0, 5.0, 5.0],
        "b": ["a", "a", "a", "d", "e", "e"],
        "c": ["a", "a", None, None, None, "f"],
        "d": [1.0, 5.0, 3.0, None, None, 1.0],
        "response": [0, 1, 0, 1, 1, 1],
        "weights_column": [0.1, 0.1, 0.8, 0.5, 0.9, 0.8],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_date_test_df(library="pandas"):
    """Create DataFrame for DateDiffLeapYearTransformer tests."""

    df_dict = {
        "a": [
            datetime.date(1993, 9, 27),
            datetime.date(2000, 3, 19),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 12, 10),
            datetime.date(1985, 7, 23),
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
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_datediff_test_df(library="pandas"):
    """Create DataFrame for DateDifferenceTransformer tests."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(2020, 5, 1, 12, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(2018, 9, 10, 9, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


# Create between dates df staying here as used in conftest also
def create_is_between_dates_df_1(library="pandas"):
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 2 datetime values."""

    df_dict = {
        "a": pd.date_range(start="1/1/2016", end="27/02/2017", periods=2),
        "b": pd.date_range(start="1/2/2016", end="27/09/2017", periods=2),
        "c": pd.date_range(start="1/3/2016", end="27/04/2017", periods=2),
        "d": pd.date_range(start="1/3/2015", end="27/04/2018", periods=2),
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_is_between_dates_df_2(library="pandas"):
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 5 datatime values, covers edge cases."""

    df_dict = {
        "a": [
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
        ],
        "b": [
            datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
        ],
        "c": [
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_is_between_dates_df_3(library="pandas"):
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 5 datatime values, covers edge cases, has mixed date data types."""

    df_dict = {
        "a_date": [
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
        ],
        "b_date": [
            datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
        ],
        "c_date": [
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
        ],
        "a_datetime": [
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
        ],
        "b_datetime": [
            datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
        ],
        "c_datetime": [
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_date_diff_different_dtypes(library="pandas"):
    """Dataframe with different datetime formats"""

    df_dict = {
        "date_col_1": [
            datetime.date(1993, 9, 27),
            datetime.date(2000, 3, 19),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 12, 10),
            datetime.date(
                1985,
                7,
                23,
            ),
        ],
        "date_col_2": [
            datetime.date(2020, 5, 1),
            datetime.date(2019, 12, 25),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 9, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 7, 23),
        ],
        "datetime_col_1": [
            datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
            datetime.datetime(2000, 3, 19, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 12, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(
                1985,
                7,
                23,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "datetime_col_2": [
            datetime.datetime(2020, 5, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2019, 12, 25, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 9, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 7, 23, tzinfo=datetime.timezone.utc),
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def create_date_diff_different_dtypes_and_nans(library="pandas"):
    """Dataframe with different datetime formats with nans in the data"""

    df_dict = {
        "date_col_1": [
            None,
            datetime.date(2000, 3, 19),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 10, 10),
            datetime.date(2018, 12, 10),
            datetime.date(
                1985,
                7,
                23,
            ),
        ],
        "date_col_2": [
            datetime.date(2020, 5, 1),
            datetime.date(2019, 12, 25),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 11, 10),
            datetime.date(2018, 9, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 11, 10),
            datetime.date(2015, 7, 23),
        ],
        "datetime_col_1": [
            datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
            datetime.datetime(2000, 3, 19, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 12, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(
                1985,
                7,
                23,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "datetime_col_2": [
            None,
            datetime.datetime(2019, 12, 25, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 9, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 7, 23, tzinfo=datetime.timezone.utc),
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)


def expected_date_diff_df_2(library="pandas"):
    """Expected output for test_expected_output_drop_cols_true."""

    df_dict = {
        "c": [
            None,
            19,
            0,
            0,
            0,
            -2,
            -3,
            30,
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library=library)
