"""Stateless nominal transforms."""

from typing import Any, Literal

import narwhals as nw
import numpy as np
from beartype import beartype


@beartype
def one_hot_encode_columns(
    columns: list[str], categories: dict[str, list[Any]], separator: str
) -> list[nw.Expr]:
    """One hot encode columns for provided categories.

    Parameters
    ----------
    columns:
        columns to set to provided value

    categories:
        dict of categories to look for per column (column:categories)

    separator:
        character to separate col name and category name in output columns

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        (nw.col(c) == level).alias(c + separator + str(level))
        for c in columns
        for level in categories[c]
    ]


@beartype
def numerically_encode_columns(  # noqa: PLR0917, PLR0913
    columns: list[str],
    mappings: dict[str, dict[str, float | int]],
    unseen_levels_encodings: dict[str, float | np.float32 | np.float64 | int] | None,
    unseen_level_handling: bool | str | int | float | None,
    return_dtypes: dict[str, Literal["Float64", "Float32"]],
    column_to_encoded_columns: dict[str, list[str]],
) -> list[nw.Expr]:
    """Numerically encode columns with provided mappings.

    Parameters
    ----------
    columns:
        columns to set to provided value

    mappings:
        mappings per level for each column (column:level:value)

    unseen_levels_encodings:
        mapping values for unseen levels per encoded column

    unseen_level_handling:
        controls whether to use unseen level handling

    return_dtypes:
        return types for each encoded column

    column_to_encoded_columns:
        dict mapping columns to the output columns they produce

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        nw.col(col)
        .alias(encoded_col)
        .replace_strict(
            mappings[encoded_col],
            default=unseen_levels_encodings[encoded_col]
            if unseen_level_handling
            else None,
        )
        .cast(getattr(nw, return_dtypes[encoded_col]))
        for col in columns
        for encoded_col in column_to_encoded_columns[col]
    ]
