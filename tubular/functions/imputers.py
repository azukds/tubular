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
