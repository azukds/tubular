"""Stateless mapping transforms."""

import warnings
from typing import Any, Literal

import narwhals as nw
from beartype import beartype

from tubular._utils import _null_safe_string_cast

RETURN_DTYPES = Literal[
    "String",
    "Categorical",
    "Boolean",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
]

FLOAT_TYPES = Literal[
    "Float32",
    "Float64",
]

INT_TYPES = Literal[
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
]

INT_TYPE_NAMES = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
]

FLOAT_TYPE_NAMES = ["Float64", "Float32"]


@beartype
def _get_full_mapping_expr(
    col: str,
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
    pre_mapping_expr: nw.Expr | None = None,
) -> nw.Expr:
    """Get expression for column which will be fully mapped.

    Example, mapping string to boolean will fail for a partial mapping,
    as casting e.g. 'cat' to bool is not defined.

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

    mapping_expr = pre_mapping_expr.replace_strict(mappings[col])

    return (
        mapping_expr.fill_null(mappings_from_null[col])
        if mappings_from_null[col] is not None
        else mapping_expr
    )


@beartype
def _get_partial_mapping_expr(
    col: str,
    mappings: dict[str, dict[Any, Any]],
    mappings_from_null: dict[str, Any],
    pre_mapping_expr: nw.Expr | None = None,
    fix_nans: bool = False,
) -> nw.Expr:
    """Get expression for column which may be partially mapped.

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

    fix_nans:
        whether to convert nan->none during the mapping process

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if pre_mapping_expr is None:
        pre_mapping_expr = nw.col(col)

    mappable_condition = nw.col(col).is_in(mappings[col])

    if fix_nans:
        mappable_condition |= nw.col(col).is_null()

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


def map_integer_columns(
    cols: list[str],
    mappings: dict[str, dict[int, Any]],
    mappings_from_null: dict[str, Any],
    return_dtypes: dict[str, RETURN_DTYPES],
    verbose: bool = True,
) -> list[nw.Expr]:
    """Get expressions for mapping int type columns.

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

    return_dtypes:
        Dictionary of col:dtype for returned columns

    verbose:
        Controls verbosity of function.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if verbose and "Boolean" in return_dtypes.values():
        warnings.warn(
            """map_integer_columns: Note if working in pandas and casting to Boolean,
            expressions output by this function are only intended for use on nullable
            type columns, as non-nullable types will result in use of the non-nullable
            'bool' type which may corrupt null values.
            This warning can be silenced by setting verbose=False.""",
            stacklevel=2,
        )

    return [
        _get_partial_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        if (return_dtypes[col] in {*INT_TYPE_NAMES, *FLOAT_TYPE_NAMES, "Categorical"})
        else _null_safe_string_cast(
            _get_partial_mapping_expr(
                col,
                mappings=mappings,
                mappings_from_null=mappings_from_null,
            )
        )
        if (return_dtypes[col] == "String")
        # boolean case
        else _get_full_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(nw.Boolean)
        for col in cols
    ]


def map_float_columns(
    cols: list[str],
    mappings: dict[str, dict[float, Any]],
    mappings_from_null: dict[str, Any],
    return_dtypes: dict[str, RETURN_DTYPES],
    verbose: bool = True,
) -> list[nw.Expr]:
    """Get expressions for mapping float type columns.

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

    return_dtypes:
        Dictionary of col:dtype for returned columns

    verbose:
        Controls the verbosity of this function.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if verbose and "Boolean" in return_dtypes.values():
        warnings.warn(
            """map_float_columns: Note if working in pandas and casting to Boolean,
            expressions output by this function are only intended for use on nullable
            type columns, as non-nullable types will result in use of the non-nullable
            'bool' type which may corrupt null values.
            This warning can be silenced by setting verbose=False.""",
            stacklevel=2,
        )

    return [
        _get_partial_mapping_expr(
            col,
            pre_mapping_expr=nw.col(col).fill_nan(None),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
            fix_nans=True,
        ).cast(getattr(nw, return_dtypes[col]))
        if (return_dtypes[col] in {*INT_TYPE_NAMES, "Categorical"})
        else _null_safe_string_cast(
            _get_partial_mapping_expr(
                col,
                pre_mapping_expr=nw.col(col).fill_nan(None),
                mappings=mappings,
                mappings_from_null=mappings_from_null,
                fix_nans=True,
            ),
        )
        if return_dtypes[col] == "String"
        else _get_partial_mapping_expr(
            col,
            pre_mapping_expr=nw.col(col),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        if return_dtypes[col] in FLOAT_TYPE_NAMES
        # boolean case
        else _get_full_mapping_expr(
            col,
            pre_mapping_expr=nw.col(col).fill_nan(None),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(nw.Boolean)
        for col in cols
    ]


def map_string_columns(
    cols: list[str],
    mappings: dict[str, dict[str, Any]],
    mappings_from_null: dict[str, Any],
    return_dtypes: dict[str, RETURN_DTYPES],
    verbose: bool = True,
) -> list[nw.Expr]:
    """Get expressions for mapping string type columns.

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

    return_dtypes:
        Dictionary of col:dtype for returned columns

    verbose:
        Controls verbosity of function.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if verbose and "Boolean" in return_dtypes.values():
        warnings.warn(
            """map_string_columns: Note if working in pandas and casting to Boolean,
            expressions output by this function are only intended for use on nullable
            type columns, as non-nullable types will result in use of the non-nullable
            'bool' type which may corrupt null values.
            This warning can be silenced by setting verbose=False.""",
            stacklevel=2,
        )

    return [
        _get_partial_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        if (return_dtypes[col] == "Categorical")
        else _null_safe_string_cast(
            _get_partial_mapping_expr(
                col,
                mappings=mappings,
                mappings_from_null=mappings_from_null,
            )
        )
        if (return_dtypes[col] == "String")
        # boolean, int, float cases
        else _get_full_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        for col in cols
    ]


def map_categorical_columns(
    cols: list[str],
    mappings: dict[str, dict[str, Any]],
    mappings_from_null: dict[str, Any],
    return_dtypes: dict[str, RETURN_DTYPES],
    verbose: bool = True,
) -> list[nw.Expr]:
    """Get expressions for mapping categorical type columns.

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

    return_dtypes:
        Dictionary of col:dtype for returned columns

    verbose:
        controls verbosity of function.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if verbose and "Boolean" in return_dtypes.values():
        warnings.warn(
            """map_categorical_columns: Note if working in pandas, it is
            not recommended to use this function to cast to Boolean, as
            result will be 'bool' type and may corrupt null values.
            This warning can be silenced by setting verbose=False.""",
            stacklevel=2,
        )

    return [
        _get_partial_mapping_expr(
            col,
            pre_mapping_expr=_null_safe_string_cast(nw.col(col)),
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        if (return_dtypes[col] == "Categorical")
        else _null_safe_string_cast(
            _get_partial_mapping_expr(
                col,
                pre_mapping_expr=_null_safe_string_cast(nw.col(col)),
                mappings=mappings,
                mappings_from_null=mappings_from_null,
            )
        )
        if (return_dtypes[col] == "String")
        # boolean, int, float cases
        else _get_full_mapping_expr(
            col,
            mappings=mappings,
            pre_mapping_expr=_null_safe_string_cast(nw.col(col)),
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        for col in cols
    ]


def map_boolean_columns(
    cols: list[str],
    mappings: dict[str, dict[bool, Any]],
    mappings_from_null: dict[str, Any],
    return_dtypes: dict[str, RETURN_DTYPES],
    verbose: bool = True,
) -> list[nw.Expr]:
    """Get expressions for mapping boolean type columns.

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

    return_dtypes:
        Dictionary of col:dtype for returned columns

    verbose:
        Control verbosity of function.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    if verbose:
        warnings.warn(
            """map_boolean_columns: Note if working in pandas and casting to/from Boolean,
            expressions output by this function are only intended for use on nullable
            type columns, as non-nullable types will result in use of the non-nullable
            'bool' type which may corrupt null values.
            This warning can be silenced by setting verbose=False.""",
            stacklevel=2,
        )

    return [
        _get_partial_mapping_expr(
            col,
            mappings=mappings,
            mappings_from_null=mappings_from_null,
        ).cast(getattr(nw, return_dtypes[col]))
        if return_dtypes[col] != "String"
        else _null_safe_string_cast(
            _get_partial_mapping_expr(
                col,
                mappings=mappings,
                mappings_from_null=mappings_from_null,
            )
        )
        for col in cols
    ]
