import pandas as pd


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
