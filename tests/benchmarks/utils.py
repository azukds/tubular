import narwhals as nw


def transform_with_exprs(exprs: list[nw.Expr], df: nw.DataFrame):

    return df.with_columns(*exprs)
