from enum import Enum
from typing import Annotated, Any

import narwhals as nw
from beartype.vale import Is


def set_columns_to_value(columns: list[str], value: Any) -> list[nw.Expr]:
    """Set columns to fixed value.

    Parameters
    ----------
    columns:
        columns to set to provided value

    value:
        value to set columns to

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [nw.lit(value).alias(c) for c in columns]


def rename_columns(
    columns: list[str], new_column_names: dict[str, str]
) -> list[nw.Expr]:
    """Rename columns with provided dictionary.

    Parameters
    ----------
    columns:
        columns to set to provided value

    new_column_names:
        dictionary of format col:new_name

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [nw.col(c).alias(new_column_names[c]) for c in columns]


class SimpleCastDtypes(str, Enum):
    """Allowed dtypes for ColumnDtypeSetter."""

    FLOAT64 = "Float64"
    FLOAT32 = "Float32"
    INT64 = "Int64"
    INT32 = "Int32"
    INT16 = "Int16"
    INT8 = "Int8"
    UINT64 = "UInt64"
    UINT32 = "UInt32"
    UINT16 = "UInt16"
    UINT8 = "UInt8"
    BOOLEAN = "Boolean"
    STRING = "String"
    CATEGORICAL = "Categorical"


SimpleCastDtypesStr = Annotated[
    str,
    Is[lambda s: s in SimpleCastDtypes._value2member_map_],
]


def cast_columns(columns: list[str], dtype: SimpleCastDtypesStr) -> list[nw.Expr]:
    """Rename columns with provided dictionary.

    Parameters
    ----------
    columns:
        columns to set to provided value

    dtype:
        dtype to cast to

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [nw.col(col).cast(getattr(nw, dtype)) for col in columns]
