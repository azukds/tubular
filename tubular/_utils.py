from typing import Literal

import narwhals as nw
import pandas as pd
from narwhals.typing import IntoDType
from numpy.typing import ArrayLike


def _assess_pandas_object_column(pandas_df: pd.DataFrame, col: str) -> tuple[str, str]:
    """tries to determine less generic type for object columns

    Parameters
    ----------
    pandas_df: pd.DataFrame
        pandas df to assess

    col: str
        column to assess

    Returns
    ----------
    pandas_col_type: str
        deduced pandas col type

    polars_col_type: str
        deduced polars col type
    """

    if pandas_df[col].dtype.name != "object":
        msg = "_assess_pandas_object_column only works with object dtype columns"
        raise TypeError(
            msg,
        )

    pandas_col_type = "object"
    polars_col_type = "Object"

    # pandas would assign dtype object to bools with nulls, but have values like True
    # it would also assign all null cols to object, but have values like None
    # creating a polars col with object type would give values like 'true', 'none'
    # overwrite these cases for better handling
    if pandas_df[col].notna().sum() == 0:
        pandas_col_type = "null"
        polars_col_type = "Unknown"

    # check if all non-null values are bool
    elif sum(isinstance(value, bool) for value in pandas_df[col] if value):
        pandas_col_type = "bool"
        polars_col_type = "Boolean"

    # we may wish to add more cases in future, but starting with these

    return pandas_col_type, polars_col_type


def new_narwhals_series_with_optimal_pandas_types(
    name: str,
    values: ArrayLike,
    backend: Literal["pandas", "polars"],
    dtype: IntoDType,
) -> nw.Series:
    """wraps around nw.new_series to ensure that pandas doesn't default to non-nullable
    types

    Parameters
    ----------
    name:
        name of new series

    values:
        values for new series

    backend:
        data processing package to use (pandas or polars)

    dtype:
        wanted narwhals dtype for output

    Returns
    ----------
    nw.Series: new narwhals series
    """

    if backend == "pandas":
        series = pd.Series(name=name, data=values)
        series = nw.maybe_convert_dtypes(nw.from_native(series, allow_series=True))
        # for int/float values, we may still want to cast to better type
        # (e.g. int8)
        # but for bool/str values, maybe_convert_types has already
        # cast to improved typing, so avoid further casting for these
        if dtype not in [nw.String, nw.Object, nw.Boolean]:
            series = series.cast(dtype)

    else:
        series = nw.new_series(name=name, values=values, backend=backend, dtype=dtype)

    return series
