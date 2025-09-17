import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular._utils import _return_narwhals_or_native_dataframe


@pytest.mark.parametrize(
    ("return_native", "df_native_type", "function_df_arg_type"),
    [
        (True, "pandas", "pandas"),
        (True, "polars", "polars"),
        (True, "pandas", "narwhals"),
        (True, "polars", "narwhals"),
        (False, "pandas", "pandas"),
        (False, "polars", "polars"),
        (False, "pandas", "narwhals"),
        (False, "polars", "narwhals"),
    ],
)
def test_outcomes(return_native, df_native_type, function_df_arg_type):
    df_dict = {
        "a": [1, 2, 3],
        "b": ["a", "b", None],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=df_native_type)

    if function_df_arg_type == "narwhals":
        df = nw.from_native(df)

    output = _return_narwhals_or_native_dataframe(df, return_native)

    if return_native and df_native_type == "pandas":
        assert isinstance(output, pd.DataFrame)

    elif return_native and df_native_type == "polars":
        assert isinstance(output, pl.DataFrame)

    elif not return_native:
        assert isinstance(output, nw.DataFrame)
        output = nw.to_native(output)

    if function_df_arg_type == "narwhals":
        df = nw.to_native(df)

    assert_frame_equal_dispatch(output, df)
