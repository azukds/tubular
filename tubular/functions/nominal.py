"""Stateless nominal transforms."""

from typing import Any, Literal

import narwhals as nw
import numpy as np
from beartype import beartype

from tubular.types import ListOfStrs


@beartype
def _get_non_rare_condition_expression(
    col: str,
    non_rare_levels: dict[str, ListOfStrs],
    unseen_levels_to_rare: bool,
    training_data_levels: dict[str, ListOfStrs] | None,
) -> nw.Expr:
    """Get expression for non rare condition.

    Parameters
    ----------
    col:
        column to map

    non_rare_levels :
        dict of non rare levels per column

    unseen_levels_to_rare:
        whether to map unseen levels to rare

    training_data_levels:
        dict of training data levels per column

    Returns
    -------
    nw.Expr: expression for non rare condition

    """
    return (
        nw.col(col).is_in(non_rare_levels[col])
        if unseen_levels_to_rare
        # if unseen levels are mapped to rare,
        # the condition becomes either in
        # non rare levels OR not in training data
        # levels (unseen)
        else (
            nw.col(col).is_in(non_rare_levels[col])
            | ~nw.col(col).is_in(training_data_levels[col])
        )
    )


@beartype
def _get_rare_grouping_expr(
    col: str,
    rare_level_name: str | ListOfStrs,
    non_rare_condition_expression: nw.Expr,
    non_rare_levels: dict,
    str_col: bool,
) -> nw.Expr:
    """Get expression for mapping column.

    Parameters
    ----------
    col:
        column to map

    rare_level_name:
        name to use for rare levels

    non_rare_condition_expression
        expression for non rare condition

    non_rare_levels:
        dict of non rare levels per column

    str_col:
        whether the column is a string column

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    pre_grouping_expr = nw.col(col) if str_col else nw.col(col).cast(nw.String)

    transform_expression = (
        nw.when(non_rare_condition_expression | nw.col(col).is_null())
        .then(pre_grouping_expr)
        .otherwise(nw.lit(rare_level_name))
    )

    if not str_col:
        transform_expression = transform_expression.cast(
            nw.Enum(non_rare_levels[col] + [rare_level_name]),
        )

    return transform_expression


@beartype
def rare_encode_categorical_or_enum_columns(
    cols: list[str],
    non_rare_levels: dict[str, ListOfStrs],
    unseen_levels_to_rare: bool,
    training_data_levels: dict[str, ListOfStrs] | None,
    rare_level_name: str | ListOfStrs,
) -> list[nw.Expr]:
    """Get expression for applying rare grouping to categorical or enum columns.

    Parameters
    ----------
    cols:
        columns to map

    non_rare_levels :
        dict of non rare levels per column

    unseen_levels_to_rare:
        whether to map unseen levels to rare

    training_data_levels:
        dict of training data levels per column

    rare_level_name:
        name to use for rare levels

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    grouped_expressions = []
    for col in cols:
        non_rare_condition_expression = _get_non_rare_condition_expression(
            col,
            non_rare_levels=non_rare_levels,
            unseen_levels_to_rare=unseen_levels_to_rare,
            training_data_levels=training_data_levels,
        )
        grouped_expressions.append(
            _get_rare_grouping_expr(
                col,
                rare_level_name=rare_level_name,
                non_rare_condition_expression=non_rare_condition_expression,
                non_rare_levels=non_rare_levels,
                str_col=False,
            )
        )
    return grouped_expressions


def rare_encode_str_columns(
    cols: list[str],
    non_rare_levels: dict,
    unseen_levels_to_rare: bool,
    training_data_levels: dict[set] | None,
    rare_level_name: str | ListOfStrs,
) -> list[nw.Expr]:
    """Get expression for applying rare grouping to string columns.

    Parameters
    ----------
    cols:
        columns to map

    non_rare_levels :
        dict of non rare levels per column

    unseen_levels_to_rare:
        whether to map unseen levels to rare

    training_data_levels:
        dict of training data levels per column

    rare_level_name:
        name to use for rare levels

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    grouped_expressions = []
    for col in cols:
        non_rare_condition_expression = _get_non_rare_condition_expression(
            col,
            non_rare_levels=non_rare_levels,
            unseen_levels_to_rare=unseen_levels_to_rare,
            training_data_levels=training_data_levels,
        )
        grouped_expressions.append(
            _get_rare_grouping_expr(
                col,
                rare_level_name=rare_level_name,
                non_rare_condition_expression=non_rare_condition_expression,
                non_rare_levels=non_rare_levels,
                str_col=True,
            )
        )
    return grouped_expressions


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
