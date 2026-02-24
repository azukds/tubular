import narwhals as nw

from tubular.types import DataFrame


def _get_null_filter_expr(column: str) -> dict[str, nw.Expr]:
    """Get expressions to filter out null rows in given columns.

    Parameters
    ----------
    column: str
        column to filter

    Returns
    -------
    nw.Expr: column null filter expression

    """
    return nw.col(column).is_null()


def _get_all_null_columns(
    X: DataFrame,
    columns: list[str],
) -> list[str]:
    """Find columns in provided dataframe which are all null.

    Parameters
    ----------
    X : DataFrame
        dataframe to check

    columns: list[str]
        list of columns in dataframe to check

    Returns
    -------
    list[str]: list of all null columns

    """
    null_exprs = {c: nw.col(c).is_null().all() for c in columns}

    null_results = X.select(**null_exprs).to_dict(as_series=False)

    return [col for col in columns if null_results[col][0] is True]
