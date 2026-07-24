"""Stateless mapping transforms."""

from typing import Any, Literal

import narwhals as nw
from beartype import beartype

RETURN_DTYPES = Literal[
    "String",
    "Categorical",
    "Boolean",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
]


@beartype
def _get_mapping_expr(
    col: str,
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
    pre_mapping_expr: nw.Expr | None = None,
) -> nw.Expr:
    """Get expression for mapping column.

    Parameters
    ----------
    col:
        column to map

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    pre_mapping_expr:
        expression containing any transforms necessary prior to mapping logic.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if pre_mapping_expr is None:
        pre_mapping_expr = nw.col(col)

    mappable_condition = nw.col(col).is_in(mappings[col])

    mapping_expr = (
        nw.when(mappable_condition)
        .then(
            # default here allows replace_strict to work, but the nulls are replaced
            # in the otherwise section anyway
            pre_mapping_expr.replace_strict(mappings[col], default=None)
        )
        .otherwise(pre_mapping_expr)
    )

    return (
        mapping_expr.fill_null(mappings_from_null[col])
        if mappings_from_null[col] is not None
        else mapping_expr
    )


@beartype
def map_number_to_string(
    cols: list[str],
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
) -> list[nw.Expr]:
    """Get expression for mapping numeric columns into string type columns.

    Parameters
    ----------
    cols:
        columns to map

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        _get_mapping_expr(
            col,
            pre_mapping_expr=nw.col(col).fill_nan(None),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        )
        for col in cols
    ]


@beartype
def map_generic_to_bool(
    cols: list[str],
    library: Literal["pandas", "polars"],
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
) -> list[nw.Expr]:
    """Get expression for mapping columns into boolean type columns.

    Parameters
    ----------
    cols:
        columns to map

    library:
        pandas or polars.

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        _get_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(nw.Boolean)
        if library == "polars"
        else _get_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        )
        for col in cols
    ]


@beartype
def map_generic_to_str(
    cols: list[str],
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
) -> list[nw.Expr]:
    """Get expression for mapping non-numeric columns into string type columns.

    Parameters
    ----------
    cols:
        columns to map

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        _get_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(nw.String)
        for col in cols
    ]


@beartype
def map_generic_to_number(
    cols: list[str],
    return_dtypes: dict[str, RETURN_DTYPES],
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
) -> list[nw.Expr]:
    """Get expression for mapping generic type columns into numeric type columns.

    Parameters
    ----------
    cols:
        columns to map

    return_dtypes:
        Dictionary of col:dtype for returned columns

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        _get_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        for col in cols
    ]


@beartype
def map_generic_to_categorical(
    cols: list[str],
    return_dtypes: dict[str, RETURN_DTYPES],
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
) -> list[nw.Expr]:
    """Get expression for mapping generic type columns into categorical type columns.

    Parameters
    ----------
    cols:
        columns to map

    return_dtypes:
        Dictionary of col:dtype for returned columns

    mappings :
        Dictionary of mappings for each column individually. The dict passed to mappings in
    init is set to the mappings attribute.

    mappings_from_null:
        dict storing what null values will be mapped to. Generally best to use an imputer,
    but this functionality is useful for inverting pipelines.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        _get_mapping_expr(
            col,
            pre_mapping_expr=nw.col(col).cast(nw.String),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        for col in cols
    ]
