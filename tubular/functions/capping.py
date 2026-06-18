"""Contains stateless transforms for capping."""

from typing import Annotated

import narwhals as nw
from beartype import beartype
from beartype.vale import Is

from tubular.types import FloatTypeAnnotated, Number

CappingValues = Annotated[
    list[Number | None],
    Is[
        lambda list_arg: (
            (len(list_arg) == 2)  # noqa: PLR2004
            & (
                all(
                    (isinstance(value, (int, float)) or value is None)
                    for value in list_arg
                )
            )
        )
    ],
]


@beartype
def cap_columns(
    columns: list[str], column_capping_ranges: dict[str, CappingValues]
) -> list[nw.Expr]:
    """Get expression for capping columns within provided ranges.

    Parameters
    ----------
    columns:
        columns to cap

    column_capping_ranges:
        dict containing per column capping ranges, in format {col: [upper_bound, lower_bound]}

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        nw.col(col).clip(
            lower_bound=column_capping_ranges[col][0],
            upper_bound=column_capping_ranges[col][1],
        )
        for col in columns
    ]


@beartype
def set_out_of_range_to_none(
    columns: list[str],
    column_capping_ranges: dict[str, CappingValues],
    dtype: FloatTypeAnnotated,
) -> list[nw.Expr]:
    """Get expression for mapping column values outside of provided range to None.

    Parameters
    ----------
    columns:
        columns to cap

    column_capping_ranges:
        dict containing per column capping ranges, in format {col: [upper_bound, lower_bound]}

    dtype:
        column dtype to return

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        nw.when(
            (nw.col(col) >= (column_capping_ranges[col][0]))
            & (nw.col(col) <= (column_capping_ranges[col][1]))
        )
        .then(nw.col(col))
        .otherwise(None)
        .cast(getattr(nw, dtype))
        .alias(col)
        if (
            column_capping_ranges[col][0] is not None
            and column_capping_ranges[col][1] is not None
        )
        else nw.when(nw.col(col) < column_capping_ranges[col][0])
        .then(None)
        .otherwise(nw.col(col))
        .cast(getattr(nw, dtype))
        .alias(col)
        if (column_capping_ranges[col][0] is not None)
        else nw.when(nw.col(col) > column_capping_ranges[col][1])
        .then(None)
        .otherwise(nw.col(col))
        .cast(getattr(nw, dtype))
        .alias(col)
        if (column_capping_ranges[col][1] is not None)
        else nw.col(col).cast(getattr(nw, dtype)).alias(col)
        for col in columns
    ]
