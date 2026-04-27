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
