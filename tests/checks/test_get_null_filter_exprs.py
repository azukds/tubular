import narwhals as nw
import pytest

from tests.utils import dataframe_init_dispatch
from tubular._checks import _get_null_filter_exprs


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_output(library):
    "test simple output case for function"

    df_dict = {"a": [1, 2, None], "b": [None, None, None], "c": [1, 2, 3]}

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    cols = ["a", "b", "c"]

    exprs = _get_null_filter_exprs(cols)

    expected_outputs = {"a": [1, 2], "b": [], "c": [1, 2, 3]}

    df = nw.from_native(df)
    for col in cols:
        output = df.select(nw.col(col)).filter(~exprs[col]).to_dict()

        assert output[col].to_list() == expected_outputs[col]
