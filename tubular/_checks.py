import narwhals as nw

from tubular.types import DataFrame


def _get_null_filter_exprs(columns: list[str]) -> dict[str, nw.Expr]:
    """Get expressions to filter out null rows in given columns.

    Parameters
    ----------
    columns: list[str]
        list of columns in to filter

    Returns
    -------
    dict[str, nw.Expr]: dict of per column null filter expressions

    """
    return {col: nw.col(col).is_null() for col in columns}


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
