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


def extract_string_components(
    columns: list[str], by: str, return_n_components: int
) -> list[nw.Expr]:
    """Get expression for extracting components from a str columns, split by provided character.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        nw.col(col).str.split(by=by).list.get(i).alias(f"{col}_split_by_{by}_entry_{i}")
        for col in columns
        for i in range(return_n_components)
    ]
