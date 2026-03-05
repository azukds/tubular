import narwhals as nw
import pytest

from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular._utils import _collect_frame


@pytest.mark.parametrize("library", ["polars"])
def test_lazyframes_collected(library):
    "test lazyframes collected by function"

    df_dict = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)

    output = _collect_frame(df.lazy())

    assert isinstance(
        output,
        nw.DataFrame,
    ), "df has not been converted to narwhals as expected"

    assert_frame_equal_dispatch(output.to_native(), df.to_native())


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_eager_frame_left_alone(library):
    "test eager pandas and polars dfs are unchanged by _collect_frame"

    df_dict = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)

    output = _collect_frame(df)

    assert isinstance(
        output,
        nw.DataFrame,
    ), "df has not been converted to narwhals as expected"

    assert_frame_equal_dispatch(output.to_native(), df.to_native())
