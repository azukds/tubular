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


def impute_numeric_nulls(
    columns: ListOfStrs | list, impute_values: dict
) -> list[nw.Expr]:
    """Return expressions to impute null values for numeric columns.

    Parameters
    ----------
    columns: list
        Columns to impute.
    impute_values: dict
        Mapping of column names to imputation values. If a value is None,
        the original column expression is returned unchanged.

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
