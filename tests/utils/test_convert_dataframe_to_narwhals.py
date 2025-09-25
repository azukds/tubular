import narwhals as nw
import pytest

from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular._utils import _convert_dataframe_to_narwhals


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_narwhalification(library):
    "test pandas and polars dfs narwhalified by function"

    df_dict = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    output = _convert_dataframe_to_narwhals(df)

    assert isinstance(
        output,
        nw.DataFrame,
    ), "df has not been converted to narwhals as expected"

    assert_frame_equal_dispatch(output.to_native(), df)


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_narwhals_frame_left_alone(library):
    "test pandas and polars dfs narwhalified by function"

    df_dict = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)

    output = _convert_dataframe_to_narwhals(df)

    assert isinstance(
        output,
        nw.DataFrame,
    ), "df has not been converted to narwhals as expected"

    assert_frame_equal_dispatch(output.to_native(), df.to_native())
