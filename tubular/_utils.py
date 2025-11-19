from contextlib import suppress
from functools import wraps
from importlib.metadata import version
from typing import Literal, Optional

import narwhals as nw
import pandas as pd
import polars as pl
from beartype import beartype
from narwhals.typing import IntoDType
from numpy.typing import ArrayLike

from tubular.types import DataFrame, NarwhalsFrame, Series


@beartype
def _convert_dataframe_to_narwhals(X: DataFrame) -> NarwhalsFrame:
    """Narwhalifies dataframe, if dataframe is not already narwhals.

    Parameters
    ----------
    X: pd/pl/nw.Series
        DataFrame to narwhalify if pd/pl type

    Returns
    -------
    nw.DataFrame/LazyFrame: narwhalified dataframe

    """
    if not isinstance(X, (nw.DataFrame, nw.LazyFrame)):
        X = nw.from_native(X)

    return X


@beartype
def _convert_series_to_narwhals(y: Optional[Series] = None) -> Optional[nw.Series]:
    """Narwhalifies series, if series is not already narwhals.

    Parameters
    ----------
    y: pd/pl/nw.Series
        series to narwhalify if pd/pl type

    Returns
    -------
    nw.Series: narwhalified series

    """
    if y is not None and not isinstance(y, nw.Series):
        y = nw.from_native(y, allow_series=True)

    return y


@beartype
def _return_narwhals_or_native_dataframe(
    X: DataFrame,
    return_native: bool,
) -> DataFrame:
    """Narwhalifies series, if series is not already narwhals.

    Parameters
    ----------
    X: pd/pl/nw.DataFrame
        DataFrame to process and return

    return_native: bool
        whether to return narwhals or native dataframe/lazyframe

    Returns
    -------
    DataFrame: processed dataframe in correct type

    """
    if return_native:
        if isinstance(X, (nw.DataFrame, nw.LazyFrame)):
            return X.to_native()

        # type hint and beartype means we don't have to check here,
        # will be pandas or polars  frame
        return X

    if isinstance(X, (pl.DataFrame, pd.DataFrame, pl.LazyFrame)):
        return nw.from_native(X)

    # type hint and beartype means we don't have to check here,
    # will be pandas or polars  frame
    return X


def _assess_pandas_object_column(pandas_df: pd.DataFrame, col: str) -> tuple[str, str]:
    """Determine less generic type for object columns.

    Parameters
    ----------
    pandas_df: pd.DataFrame
        pandas df to assess

    col: str
        column to assess

    Returns
    -------
    pandas_col_type: str
        deduced pandas col type

    polars_col_type: str
        deduced polars col type

    Raises
    ------
    TypeError: if provided column is not object type

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
    """Wrap around nw.new_series to ensure that pandas doesn't default to non-nullable types.

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
    -------
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


def _get_version() -> str:
    """Dynamically retrieve package version.

    Returns
    -------
        str: package version

    """
    with suppress(ModuleNotFoundError):
        return version("tubular")

    return "dev"


def block_from_json(method):  # noqa: ANN202, ANN001,  no annotations for generic decorator
    """Intercept method and raise a runtime error if the transformer has been initialised from_json.

    i.e. if the built_from_json attr is True.

    Returns
    -------
        Callable: wrapped method

    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202, no annotations for generic decorator
        if self.built_from_json:
            msg = "Transformers that are reconstructed from json only support .transform functionality, reinitialise a new transformer to use this method"
            raise RuntimeError(
                msg,
            )

        return method(self, *args, **kwargs)

    return wrapper
