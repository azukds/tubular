import narwhals as nw


def transform_with_exprs(exprs: list[nw.Expr], df: nw.DataFrame) -> nw.DataFrame:
    r"""Benchmarkable function for expression based pipeline transforms.

    Parameters
    ----------
    exprs:
        list of transform exprs

    df:
        dataframe to transform

    Returns
    -------
        nw.DataFrame:
            transformed dataframe
    """

    return df.with_columns(*exprs)
