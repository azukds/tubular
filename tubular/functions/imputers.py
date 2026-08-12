"""Contains stateless transforms for imputing columns."""

import narwhals as nw

from tubular.types import ListOfStrs


def indicate_nulls_for_columns(columns: ListOfStrs | str) -> list[nw.Expr]:
    """Return the positions of null values for each column.

    Parameters
    ----------
    columns: str or list
        Columns to produce indicator columns for

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [(nw.col(c).is_null()).alias(f"{c}_nulls") for c in columns]


def impute_numeric_or_string_nulls(
    columns: ListOfStrs | list, impute_values: dict[str, int | float | str]
) -> list[nw.Expr]:
    """Return expressions to impute null values for numeric or string columns.

    Parameters
    ----------
    columns: list
        Columns to impute.
    impute_values: dict
        Mapping of column names to imputation values.

    Returns
    -------
    list[nw.Expr]
        Transform expressions with nulls filled for the specified columns.

    """
    return [
        nw.col(col).fill_null(value=impute_values[col])
        if (impute_values[col] is not None)
        else nw.col(col)
        for col in columns
    ]


def impute_boolean_columns(
    columns: list[str], impute_values: dict[str, bool]
) -> list[nw.Expr]:
    """Impute boolean columns with provided values.

    Parameters
    ----------
    columns:
        columns to impute

    impute_values:
        values to impute columns with

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        nw.col(col).fill_null(value=impute_values[col]).cast(nw.Boolean)
        if (impute_values[col] is not None)
        else nw.col(col)
        for col in columns
    ]
