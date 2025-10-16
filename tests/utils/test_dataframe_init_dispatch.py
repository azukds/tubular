import datetime

import numpy as np
import pandas as pd
import polars as pl

from tests.utils import dataframe_init_dispatch


def test_output_types():
    "simple sanity check that requested df type is returned"

    df_dict = {"a": [1, 2], "b": [3, 4]}

    pandas_output = dataframe_init_dispatch(library="pandas", dataframe_dict=df_dict)

    assert isinstance(
        pandas_output,
        pd.DataFrame,
    ), "dataframe_init_dispatch not returning pandas df for library=pandas"

    polars_output = dataframe_init_dispatch(library="polars", dataframe_dict=df_dict)

    assert isinstance(
        polars_output,
        pl.DataFrame,
    ), "dataframe_init_dispatch not returning polars df for library=polars"


def test_type_alignment():
    "test that types are matched as expected between polars/pandas requests"

    df_dict = {
        # int
        "a": [1, 2],
        # int with null (float)
        "a1": [1, None],
        # float
        "b": [3.0, 4.0],
        # float with None
        "c": [3.0, np.nan],
        # str
        "d": ["a", "b"],
        # str with None
        "e": ["a", None],
        # bool
        "f": [True, False],
        # bool with None
        "g": [True, None],
        # None
        "h": [None, None],
        # date
        "i": [datetime.date(2020, 5, 1), datetime.date(2021, 5, 1)],
        # datetime
        "j": [
            datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
            datetime.datetime(1995, 9, 27, tzinfo=datetime.timezone.utc),
        ],
    }

    expected_pandas_types = {
        "a": np.int64,
        "a1": np.float64,
        "b": np.float64,
        "c": np.float64,
        "d": object,
        "e": object,
        "f": bool,
        "g": object,
        "h": object,
        "i": "date32[day][pyarrow]",
        "j": "datetime64[ns, UTC]",
    }

    expected_polars_types = {
        "a": pl.Int64,
        "a1": pl.Float64,  # function is designed to copy pandas handling
        "b": pl.Float64,
        "c": pl.Float64,
        "d": pl.String,
        "e": pl.String,
        "f": pl.Boolean,
        "g": pl.Boolean,
        "h": pl.Null,
        "i": pl.Date,
        "j": pl.Datetime,
    }

    pandas_output = dataframe_init_dispatch(library="pandas", dataframe_dict=df_dict)

    polars_output = dataframe_init_dispatch(library="polars", dataframe_dict=df_dict)

    assert set(polars_output.columns) == set(pandas_output.columns), (
        "dataframe_init_dispatch: polars and pandas output should have same columns for same input dataframe_dict"
    )

    for col in pandas_output:
        assert pandas_output[col].dtype == expected_pandas_types[col]

        assert polars_output[col].dtype == expected_polars_types[col]
