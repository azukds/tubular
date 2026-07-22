from typing import Any

import narwhals as nw


def impute_string_and_numeric_columns(
    columns: list[str], impute_values: dict[str, Any]
) -> list[nw.Expr]:
    """Impute string and numeric columns with provided values.

    Parameters
    ----------
    columns:
        columns to impute

    impute_values:
        values to impute columns with

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        nw.col(col).fill_null(value=impute_values[col])
        if (impute_values[col] is not None)
        else nw.col(col)
        for col in columns
    ]


def impute_boolean_columns(
    columns: list[str], impute_values: dict[str, Any]
) -> list[nw.Expr]:
    """Impute boolean columns with provided values.

    Parameters
    ----------
    columns:
        columns to impute

    impute_values:
        values to impute columns with

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        nw.col(col).fill_null(value=impute_values[col]).cast(nw.Boolean)
        if (impute_values[col] is not None)
        else nw.col(col)
        for col in columns
    ]


def impute_categorical_columns(
    columns: list[str], impute_values: dict[str, Any]
) -> list[nw.Expr]:
    """Impute categorical columns with provided values.

    Parameters
    ----------
    columns:
        columns to impute

    impute_values:
        values to impute columns with

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [
        nw.when(nw.col(col).is_null())
        .then(None)
        .otherwise(nw.col(col).cast(nw.String))
        .fill_null(value=impute_values[col])
        .cast(nw.Categorical)
        .alias(col)
        if (impute_values[col] is not None)
        else nw.col(col)
        for col in columns
    ]


def indicate_nulls(columns: list[str]):
    """Create indicator columns for null values in given columns.

    Parameters
    ----------
    columns:
        columns to indicate nulls on.

    Returns
    -------
    list[nw.Expr]: transform expressions

    """
    return [nw.col(c).is_null().alias(f"{c}_nulls") for c in columns]
