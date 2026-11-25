import narwhals as nw
import numpy as np
import pytest

from tests.utils import (
    _collect_frame,
    _convert_to_lazy,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular._utils import _null_safe_string_cast


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    ("input_column", "expected", "handle_nans"),
    [
        ([1.0, None, 2.0], ["1.0", None, "2.0"], True),
        ([1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"], True),
        ([1.0, None, np.nan], ["1.0", None, None], True),
        ([1, 2, 3], ["1", "2", "3"], False),
    ],
)
def test_output(input_column, expected, library, lazy, handle_nans):
    "test basic output cases."

    df_dict = {"a": input_column}
    expected_df_dict = {"a": expected}

    if library == "pandas" and lazy:
        return

    df = dataframe_init_dispatch(df_dict, library)
    expected_df = dataframe_init_dispatch(expected_df_dict, library)

    expr = _null_safe_string_cast(nw.col("a"), handle_nans=handle_nans)

    df = _convert_to_lazy(df, lazy)
    df = nw.from_native(df)
    output = df.with_columns(expr).to_native()
    output = _collect_frame(output, lazy)

    assert_frame_equal_dispatch(expected_df, output)
