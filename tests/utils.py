import narwhals as nw
import pandas as pd
import polars as pl
from narwhals.typing import FrameT
from pandas.testing import assert_frame_equal as assert_pandas_frame_equal
from polars.testing import assert_frame_equal as assert_polars_frame_equal

from tubular._utils import (
    _assess_pandas_object_column,
)
from tubular.base import BaseTransformer
from tubular.types import DataFrame

PANDAS_TO_POLARS_TYPES = {
    "int64": pl.Int64,
    "int32": pl.Int32,
    "int16": pl.Int16,
    "int8": pl.Int8,
    "float64": pl.Float64,
    "float32": pl.Float32,
    "object": pl.Utf8,
    "str": pl.String,
    "bool": pl.Boolean,
    "datetime64[ns]": pl.Datetime,
    "datetime64[ns, UTC]": pl.Datetime(time_zone="UTC"),
    "date32[day][pyarrow]": pl.Date,
    # this is not a pandas type, but include to help manage null column handling
    "null": pl.Unknown,
}


def _align_pandas_and_polars_dtypes(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
) -> pl.DataFrame:
    """helper to ensure that dtypes are aligned between polars/pandas dfs used in tests. Avoids e.g.
    [0, None] being inferred as float in pandas and int in polars and complicating tests

    Parameters
    ----------
    pandas_df : pd.DataFrame
        pandas df to extract types from

    polars_df : pl.DataFrame
        polars df to apply types to

    Returns
    ----------

    pl.DataFrame: polars df with types converted to align with pandas_df

    """

    for col in pandas_df:
        pandas_col_type = str(pandas_df[col].dtype)

        # pandas would assign dtype object to bools with nulls, but have values like True
        # it would also assign all null cols to object, but have values like None
        # creating a polars col with object type would give values like 'true', 'none'
        # overwrite these cases for better handling
        if pandas_col_type == "object":
            pandas_col_type, polars_col_type = _assess_pandas_object_column(
                pandas_df=pandas_df,
                col=col,
            )

        polars_col_type = PANDAS_TO_POLARS_TYPES[pandas_col_type]
        polars_df = polars_df.with_columns(polars_df[col].cast(polars_col_type))

    return polars_df


def assert_frame_equal_dispatch(df1: FrameT, df2: FrameT, **pandas_kwargs) -> None:
    """fixture to return correct pandas/polars assert_frame_equal method

    Parameters
    ----------
    df1 : pd.DataFrame
        first df for comparison

    df1 : pd.DataFrame
        second df for comparison

    pandas_kwargs:
        additional kwargs to pass to pandas assert_frame_equal

    """

    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return assert_pandas_frame_equal(df1, df2, **pandas_kwargs)

    if isinstance(df1, pl.DataFrame) and isinstance(df2, pl.DataFrame):
        return assert_polars_frame_equal(df1, df2)

    invalid_request_error = "tubular is setup to handle only pandas or polars inputs, and dfs input to this function should be from same library"
    raise ValueError(invalid_request_error)


def dataframe_init_dispatch(
    dataframe_dict: dict,
    library: str,
) -> FrameT:
    """
    Initialize a DataFrame using either Pandas or Polars library based on the specified library name.

    Parameters:
    dataframe_dict (dict): A dictionary where keys are column names and values are lists of column data.
    library (str): The name of the library to use for DataFrame creation. Should be either "pandas" or "polars".

    Returns:
    pl.DataFrame | pd.DataFrame: A DataFrame object from the specified library.

    Raises:
    ValueError: If the `library` parameter is not "pandas" or "polars".
    """

    pandas_df = pd.DataFrame(dataframe_dict)

    # narwhals only supports pyarrow date type, so convert
    for col in pandas_df:
        if pd.api.types.infer_dtype(pandas_df[col]) == "date":
            pandas_df[col] = pandas_df[col].astype("date32[pyarrow]")

        # ensure datetimes setup as expected
        if pd.api.types.is_datetime64_any_dtype(pandas_df[col]):
            pandas_df[col] = pd.to_datetime(pandas_df[col])

    if library == "pandas":
        return pandas_df

    if library == "polars":
        polars_df = pl.DataFrame(dataframe_dict)

        return _align_pandas_and_polars_dtypes(pandas_df, polars_df)

    library_error_message = (
        "The library parameter should be either 'pandas' or 'polars'."
    )
    raise ValueError(library_error_message)


def _check_if_skip_test(
    transformer: BaseTransformer,
    df: DataFrame,
    lazy: bool,
    from_json: bool = False,
) -> bool:
    """
    check for conditions under which test can be skipped:
        - is a polars test on a non polars transformer
        - is a lazy test on a non lazy transformer
        - is a lazy test on a pandas dataframe
        - is a from_json test on a non jsonable transformer

    Parameters:
    -----------
    transformer:
        transformer being tested
    df:
        dataframe being tested
    lazy:
        is the test lazy?
    from_json:
        is the test for a from_json'd transformer? Defaults to False.

    Returns
    -------
    bool:
        True if test can be skipped

    """

    polars = isinstance(df, pl.DataFrame)

    return (
        (not transformer.polars_compatible and polars)
        or (not polars and lazy)
        or (not transformer.lazyframe_compatible and lazy)
        or (not transformer.jsonable and from_json)
    )


def _convert_to_lazy(df: DataFrame, lazy: bool) -> DataFrame:
    """
    converts dataframe to lazy if dataframe is polars and test is for lazyframes

    Parameters:
    -----------
    df:
        dataframe being tested
    lazy:
        is the test lazy?

    Returns
    -------
    DataFrame:
        converted or original dataframe
    """

    polars = isinstance(df, pl.DataFrame)

    return df.lazy() if (lazy and polars) else df


def _collect_frame(df: DataFrame, lazy: bool) -> DataFrame:
    """
    collect lazyframe if type is polars and test is for lazyframes

    Parameters:
    -----------
    df:
        dataframe being tested
    lazy:
        is the test lazy?

    Returns
    -------
    DataFrame:
        converted or original dataframe
    """

    lazy = isinstance(df, (pl.LazyFrame, nw.LazyFrame))

    return df.collect() if lazy else df


def _handle_from_json(
    transformer: BaseTransformer,
    from_json: bool,
) -> BaseTransformer:
    """handle converting transformer to/from json pre-testing. Assumes
    _check_if_skip_test has already been run.

    Parameters
    ----------
    transformer: BaseTransformer
        transformer being  tested

    from_json: bool
        whether to dump to and reconstruct from json pre test

    Returns
    -------
    BaseTransformer:
        transformer ready for test
    """

    return transformer.from_json(transformer.to_json()) if from_json else transformer
