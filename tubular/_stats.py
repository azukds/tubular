from typing import Optional

import narwhals as nw


def _get_median_calculation_expression(
    column: list[str],
    weights_column: str,
    initial_column_expr: Optional[list[nw.Expr]] = None,
    initial_weights_expr: Optional[nw.Expr] = None,
) -> tuple[dict[str, nw.Expr], list[str]]:
    """produce expressions for calculating medians in provided dataframe

    Parameters
    ----------

    column: list[str]
        column to find median for

    weights_column: str
        name of weights column

    initial_column_expr: nw.Expr
        initial column expressions to build on. Defaults to None,
        and in this case nw.col(column) is taken as the initial expr

    initial_weights_expr: nw.Expr
        initial expression for weights column. Defaults to None,
        and in this case nw.col(weights_column) is taken as the initial expr

    Returns
    ----------
    median_value_exprs: dict[str, nw.Expr]
        dict of format col: expression for calculating median

    """

    if initial_column_expr is None:
        initial_column_expr = nw.col(column)

    if initial_weights_expr is None:
        initial_weights_expr = nw.col(weights_column)

    if weights_column is not None:
        cumsum_weights_expr = initial_weights_expr.cum_sum()

        median_expr = initial_column_expr.filter(
            cumsum_weights_expr >= (initial_weights_expr.sum() / 2.0),
        ).min()

    else:
        median_expr = initial_column_expr.drop_nulls().median()

    return median_expr


def _get_mean_calculation_expressions(
    columns: list[str],
    weights_column: str,
    initial_columns_exprs: Optional[list[nw.Expr]] = None,
    initial_weights_expr: Optional[nw.Expr] = None,
) -> dict[str, nw.Expr]:
    """produce expressions for calculating means in provided dataframe

    Parameters
    ----------

    columns: list[str]
        list of columns to find means for

    weights_column: str
        name of weights column

    initial_columns_exprs: dict[str, nw.Expr]
        dict containing initial column expressions to build on. Defaults to None,
        and in this case nw.col(c) is taken as the initial expr for each column c.

        This argument allows the chaining of longer expressions into calculating
        the mean, so we are not restricted to working with nw.col(c) and
        could pass e.g. (nw.col(c) * 2) if this was of interest.

    initial_weights_expr: nw.Expr
        initial expression for weights column. Defaults to None,
        and in this case nw.col(weights_column) is taken as the initial expr

        This argument allows the chaining of longer expressions into calculating
        the mean, so we are not restricted to working with nw.col(weights_column)
        and could pass e.g. (nw.col(weights_column) * 2) if this was of interest.

    Returns
    ----------
    mean_value_exprs: dict[str, nw.Expr]
        dict of format col: expression for calculating means

    """

    # if a more complex starting expression for c or weights has been passed,
    # (e.g. we may be worthing with a version of c that has been mapped)
    # use this, otherwise proceed with the base case
    # nw.col(c) and nw.col(weights_column)
    if initial_columns_exprs is None:
        initial_columns_exprs = {c: nw.col(c) for c in columns}

    if initial_weights_expr is None:
        initial_weights_expr = nw.col(weights_column)

    # for each col c, calculate total weight where c is non-null
    total_weight_expressions = {
        c: (initial_weights_expr.filter(~initial_columns_exprs[c].is_null()).sum())
        for c in columns
    }

    # for each col c, calculate total weighted c where
    # c is not null
    total_weighted_col_expressions = {
        c: ((initial_columns_exprs[c] * initial_weights_expr).drop_nulls().sum())
        for c in columns
    }

    #  for each col c, take the ratio of these and return as weighted mean
    return {
        c: (total_weighted_col_expressions[c] / total_weight_expressions[c])
        for c in columns
    }


def _get_mode_calculation_expressions(
    columns: list[str],
    weights_column: str,
    initial_columns_exprs: Optional[list[nw.Expr]] = None,
    initial_weights_expr: Optional[nw.Expr] = None,
) -> tuple[dict[str, nw.Expr], list[str]]:
    """produce expressions for calculating modes in provided dataframe

    Parameters
    ----------

    columns: list[str]
        list of columns to find modes for

    weights_column: str
        name of weights column

    initial_columns_exprs: dict[str, nw.Expr]
        dict containing initial column expressions to build on. Defaults to None,
        and in this case nw.col(c) is taken as the initial expr for each column c

    initial_weights_expr: nw.Expr
        initial expression for weights column. Defaults to None,
        and in this case nw.col(weights_column) is taken as the initial expr

    Returns
    ----------
    mode_value_exprs: dict[str, nw.Expr]
        dict of format col: expression for calculating modes

    """

    if initial_columns_exprs is None:
        initial_columns_exprs = {c: nw.col(c) for c in columns}

    if initial_weights_expr is None:
        initial_weights_expr = nw.col(weights_column)

    level_weights_exprs = {
        c: (
            nw.when(~initial_columns_exprs[c].is_null())
            .then(initial_weights_expr)
            .otherwise(None)
            .sum()
            .over(c)
        )
        for c in columns
    }

    return {
        c: (
            nw.when(level_weights_exprs[c] == level_weights_exprs[c].max())
            .then(nw.col(c))
            .otherwise(None)
        )
        for c in columns
    }
