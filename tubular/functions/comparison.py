"""Contains stateless transforms for comparing columns."""

import operator
from enum import Enum
from typing import Annotated

import narwhals as nw
from beartype.vale import Is

from tubular.types import ListOfTwoStrs


def apply_when_then_otherwise(
    columns: list[str],
    when_column: str,
    then_column: str,
) -> nw.Expr:
    """Get expression for capping columns within provided ranges.

    Parameters
    ----------
    columns:
        columns to cap

    when_column:
        name of boolean mask column

    then_column:
        name of column provididing values for if 'when' condition is met

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        nw.when(nw.col(when_column))
        .then(nw.col(then_column))
        .otherwise(nw.col(col))
        .alias(col)
        for col in columns
    ]


class ConditionEnum(Enum):
    """Enumeration of comparison conditions."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL_TO = "=="
    NOT_EQUAL_TO = "!="


ConditionEnumStr = Annotated[
    str,
    Is[lambda s: s in ConditionEnum._value2member_map_],
]

# Map the enum to the operator functions
ops_map = {
    ConditionEnum.GREATER_THAN: operator.gt,
    ConditionEnum.LESS_THAN: operator.lt,
    ConditionEnum.EQUAL_TO: operator.eq,
    ConditionEnum.NOT_EQUAL_TO: operator.ne,
}


def compare_two_columns(
    columns: ListOfTwoStrs,
    condition: ConditionEnumStr,
) -> nw.Expr:
    """Get expression for capping columns within provided ranges.

    Parameters
    ----------
    columns:
        columns to cap

    condition:
        comparison condition, e.g. "<"

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    null_filter_expr = nw.col(columns[0]).is_null() | nw.col(columns[1]).is_null()

    return (
        nw.when(~null_filter_expr)
        .then(ops_map[ConditionEnum(condition)](nw.col(columns[0]), nw.col(columns[1])))
        .otherwise(None)
        .alias(f"{columns[0]}{condition}{columns[1]}")
    )
