import narwhals as nw
import pytest
from pandas.testing import assert_series_equal as assert_pandas_series_equal
from polars.testing import assert_series_equal as assert_polars_series_equal

from tests.utils import dataframe_init_dispatch
from tubular._utils import _convert_series_to_narwhals


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_narwhalification(library):
    "test pandas and polars series narwhalified by function"

    df_dict = {
        "a": [1, 2, 3],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    series = df["a"]

    native_namespace = nw.get_native_namespace(series).__name__

    output = _convert_series_to_narwhals(series)

    assert isinstance(
        output,
        nw.Series,
    ), "series has not been converted to narwhals as expected"

    if native_namespace == "pandas":
        assert_pandas_series_equal(output.to_native(), series)

    if native_namespace == "polars":
        assert_polars_series_equal(output.to_native(), series)


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_narwhals_series_left_alone(library):
    "test pandas and polars series narwhalified by function"

    df_dict = {
        "a": [1, 2, 3],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    series = df["a"]

    series = nw.from_native(series, allow_series=True)

    output = _convert_series_to_narwhals(series)

    native_namespace = nw.get_native_namespace(series).__name__

    assert isinstance(
        output,
        nw.Series,
    ), "series has not been converted to narwhals as expected"

    if native_namespace == "pandas":
        assert_pandas_series_equal(output.to_native(), series.to_native())

    if native_namespace == "polars":
        assert_polars_series_equal(output.to_native(), series.to_native())
