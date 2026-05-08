"""Stateless numeric transforms."""

import narwhals as nw

from tubular.types import FloatTypeAnnotated, ListOfTwoStrs


def get_difference_of_two_columns(
    columns: ListOfTwoStrs,
) -> nw.Expr:
    """Get difference column[0]-column[1].

    Parameters
    ----------
    columns:
        columns to diff

    Returns
    -------
    nw.Expr: transform expressions

    """
    return (nw.col(columns[0]) - nw.col(columns[1])).alias(
        f"{columns[0]}_minus_{columns[1]}"
    )


def get_ratio_of_two_columns(
    columns: ListOfTwoStrs,
    return_dtype: FloatTypeAnnotated,
) -> nw.Expr:
    """Divide column[0] by column[1].

    Parameters
    ----------
    columns:
        columns to take ratio of

    return_dtype:
        Float64 or Float32

    Returns
    -------
    nw.Expr: transform expressions

    """
    return (
        nw.when(nw.col(columns[1]) != 0)
        .then(nw.col(columns[0]) / nw.col(columns[1]))
        .otherwise(None)
        .cast(getattr(nw, return_dtype))
        .alias(f"{columns[0]}_divided_by_{columns[1]}")
    )
