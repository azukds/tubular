"""Contains stateless transforms for data aggregations."""

from enum import Enum

import narwhals as nw
from beartype.typing import Annotated, List
from beartype.vale import Is


class ColumnsOverRowAggregationOptions(str, Enum):
    """Aggregation options for ColumnsOverRowAggregationTransformer."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    # not currently easy to implement row-wise
    # median or count, so leaving out for now


class RowsOverColumnsAggregationOptions(str, Enum):
    """Aggregation options for RowsOverColumnAggregationTransformer."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    COUNT = "count"


ListOfColumnsOverRowAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in ColumnsOverRowAggregationOptions._value2member_map_
            for entry in list_value
        )
    ],
]

ListOfRowsOverColumnsAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in RowsOverColumnsAggregationOptions._value2member_map_
            for entry in list_value
        )
    ],
]

horizontal_expr_map = {
    "min": nw.min_horizontal,
    "max": nw.max_horizontal,
    "sum": nw.sum_horizontal,
    "mean": nw.mean_horizontal,
}


def aggregate_over_rows(
    columns: list[str], key: str, aggregations: ListOfRowsOverColumnsAggregations
) -> nw.Expr:
    """Get expressions for aggregating data over rows, grouping by a key.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        getattr(nw.col(col), agg)().over(key).alias(f"{col}_{agg}")
        for col in columns
        for agg in aggregations
    ]


def aggregate_over_columns(
    columns: list[str], aggregations: ListOfColumnsOverRowAggregations
) -> nw.Expr:
    """Get expressions for aggregating data over columns.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return (
        [
            horizontal_expr_map[aggregation](columns).alias(
                "_".join(columns) + "_" + aggregation,
            )
            for aggregation in aggregations
        ]
        if columns
        else []
    )
