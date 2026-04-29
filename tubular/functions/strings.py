"""Stateless string transforms."""

import narwhals as nw


def convert_string_columns_to_lowercase(columns: list[str]) -> list[nw.Expr]:
    """Get expression for converting columns to lowercase.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [nw.col(col).str.to_lowercase() for col in columns]


def remove_characters_from_string_columns(
    columns: list[str], characters_formatted: str
) -> list[nw.Expr]:
    """Get expression for removing characters from string columns.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [nw.col(col).str.replace_all(characters_formatted, "") for col in columns]


def indicate_if_string_columns_contain_reference(
    columns: list[str], reference: str, reference_as_column: bool
) -> list[nw.Expr]:
    """Get expression for removing characters from string columns.

    Parameters
    ----------
    columns: list[str]
        list of columns to search

    reference: str
            reference value to search for

    reference_as_column: bool
        whether to treat reference as a column or a literal

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    reference_for_expr = nw.col(reference) if reference_as_column else reference

    return [
        nw.col(col)
        .str.contains(reference_for_expr)
        .alias(f"{col}_contains_{reference}")
        for col in columns
    ]
