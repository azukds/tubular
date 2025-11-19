"""Contains transformers for performing data aggregations."""

from enum import Enum
from typing import Union

import narwhals as nw
from beartype import beartype
from beartype.typing import Annotated, List, Optional
from beartype.vale import Is

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
)
from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin
from tubular.types import DataFrame, NumericTypes


class ColumnsOverRowAggregationOptions(str, Enum):
    """Aggregation options fo ColumnsOverRowAggregationTransformer."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    # not currently easy to implement row-wise
    # median or count, so leaving out for now


class RowsOverColumnsAggregationOptions(str, Enum):
    """Aggregation options fo RowsOverColumnAggregationTransformer."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    COUNT = "count"


ListOfColumnsOverRowAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in ColumnsOverRowAggregationOptions._value2member_map_
            for entry in list_value
        )
    ],
]

ListOfRowsOverColumnsAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in RowsOverColumnsAggregationOptions._value2member_map_
            for entry in list_value
        )
    ],
]


class BaseAggregationTransformer(BaseTransformer, DropOriginalMixin):
    """Base class for aggregation transformers.

    This class provides the foundation for aggregation-based transformations,
    handling common setup tasks such as validating aggregation methods and
    managing column specifications.

    Attributes:
    ----------
    columns : Union[str, list[str]]
        Columns to apply the transformations to.

    aggregations : list[str]
        Aggregation methods to apply.

    drop_original : bool
        Indicator for dropping original columns.

    verbose : bool
        Indicator for verbose output.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible: bool
        Indicates if transformer will work with polars frames

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    >>> BaseAggregationTransformer(
    ... columns='a',
    ... aggregations=['min', 'max'],
    ...    )
    BaseAggregationTransformer(aggregations=['min', 'max'], columns=['a'])

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        aggregations: Union[
            ListOfColumnsOverRowAggregations,
            ListOfRowsOverColumnsAggregations,
        ],
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialise class.

        Parameters
        ----------
        columns : list[str]
            List of column names to apply the aggregation transformations to.
        aggregations : list[str]
            List of aggregation methods to apply. Valid methods include 'min', 'max',
            'mean', 'median', and 'count'.
        drop_original : bool, optional
            Whether to drop the original columns after transformation. Default is False.
        verbose : bool, optional
            If True, enables verbose output for debugging purposes. Default is False.

        """
        super().__init__(columns=columns, verbose=verbose)

        self.aggregations = aggregations

        self.drop_original = drop_original

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Perform pre-transform safety checks.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating specified columns.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            checked dataframe to transform.

        Raises
        ------
        TypeError: If columns are non-numeric.

        Examples
        --------
            >>> import polars as pl

            >>> transformer=BaseAggregationTransformer(
            ... columns='a',
            ... aggregations=['min', 'max'],
            ...    )

            >>> test_df=pl.DataFrame({'a': [1,2], 'b': [3,4]})

            >>> # base transformers have no effect on data
            >>> transformer.transform(test_df)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘

        """
        return_native = self._process_return_native(return_native_override)

        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        schema = X.schema

        non_numerical_columns = [
            col for col in self.columns if schema[col] not in NumericTypes
        ]

        # convert to list and sort for consistency in return
        non_numerical_columns = list(non_numerical_columns)
        non_numerical_columns.sort()
        if len(non_numerical_columns) != 0:
            msg = f"{self.classname}: attempting to call transformer on non-numeric columns {non_numerical_columns}, which is not supported"
            raise TypeError(msg)

        return _return_narwhals_or_native_dataframe(X, return_native=return_native)


class AggregateRowsOverColumnTransformer(BaseAggregationTransformer):
    """Aggregate rows over specified columns, where rows are grouped by provided key column.

    Attributes:
    ----------
    columns : Union[str, list[str]]
        List of column names to apply the aggregation transformations to.

    aggregations : list[str]
        List of aggregation methods to apply.

    key : str
        Column name to group by for aggregation.

    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible: bool
        Indicates if transformer will work with polars frames

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
        >>> AggregateRowsOverColumnTransformer(
        ... columns='a',
        ... aggregations=['min', 'max'],
        ... key='b',
        ... )
        AggregateRowsOverColumnTransformer(aggregations=['min', 'max'], columns=['a'],
                                           key='b')

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        aggregations: ListOfRowsOverColumnsAggregations,
        key: str,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialise class.

        Parameters
        ----------
        columns : Union[str, list[str]]
            List of column names to apply the aggregation transformations to.

        aggregations : list[str]
            List of aggregation methods to apply.

        key : str
            Column name to group by for aggregation.

        drop_original : bool, optional
            Whether to drop the original columns after transformation. Default is False.

        verbose: bool
            Controls verbosity of transformer

        """
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            verbose=verbose,
        )
        self.key = key

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        >>> transformer  = AggregateRowsOverColumnTransformer(
        ... columns='a',
        ... aggregations=['min', 'max'],
        ... key='b',
        ... )

        >>> transformer.get_feature_names_out()
        ['a_min', 'a_max']

        """
        return [f"{col}_{agg}" for col in self.columns for agg in self.aggregations]

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the dataframe by aggregating rows over specified columns.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating specified columns.

        Returns
        -------
            pd.DataFrame or pl.DataFrame
                Transformed DataFrame with aggregated columns.

        Raises
        ------
        ValueError
            If the key column is not found in the DataFrame.

        Examples
        --------
            >>> import polars as pl

            >>> transformer=AggregateRowsOverColumnTransformer(
            ... columns='a',
            ... aggregations=['min', 'max'],
            ... key='b',
            ...    )

            >>> test_df=pl.DataFrame({'a': [1,2,3], 'b': [1,1,2], 'c':[1,2,3]})

            >>> transformer.transform(test_df)
            shape: (3, 5)
            ┌─────┬─────┬─────┬───────┬───────┐
            │ a   ┆ b   ┆ c   ┆ a_min ┆ a_max │
            │ --- ┆ --- ┆ --- ┆ ---   ┆ ---   │
            │ i64 ┆ i64 ┆ i64 ┆ i64   ┆ i64   │
            ╞═════╪═════╪═════╪═══════╪═══════╡
            │ 1   ┆ 1   ┆ 1   ┆ 1     ┆ 2     │
            │ 2   ┆ 1   ┆ 2   ┆ 1     ┆ 2     │
            │ 3   ┆ 2   ┆ 3   ┆ 3     ┆ 3     │
            └─────┴─────┴─────┴───────┴───────┘

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        if self.key not in X.columns:
            msg = f"{self.classname()}: key '{self.key}' not found in dataframe columns"
            raise ValueError(msg)

        expr_dict = {
            f"{col}_{agg}": getattr(nw.col(col), agg)().over(self.key)
            for col in self.columns
            for agg in self.aggregations
        }

        X = X.with_columns(**expr_dict)

        X = self.drop_original_column(
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )

        # Use mixin method to drop original columns
        return _return_narwhals_or_native_dataframe(X, self.return_native)


class AggregateColumnsOverRowTransformer(BaseAggregationTransformer):
    """Aggregate provided columns over each row.

    This transformer aggregates data within specified columns
    and can optionally drop the original columns post-transformation.

    Attributes:
    ----------
    columns : Union[str,list[str]]
        List of column names to apply the aggregation transformations to.

    aggregations : list[str]
        List of aggregation methods to apply.

    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.

    verbose : bool, optional
        Indicator for verbose output.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible: bool
        Indicates if transformer will work with polars frames

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    >>> AggregateColumnsOverRowTransformer(
    ... columns=['a', 'b'],
    ... aggregations=['min', 'max'],
    ... )
    AggregateColumnsOverRowTransformer(aggregations=['min', 'max'],
                                       columns=['a', 'b'])

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        aggregations: ListOfColumnsOverRowAggregations,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialise class.

        Parameters
        ----------
        columns : Union[str,list[str]]
            List of column names to apply the aggregation transformations to.

        aggregations : list[str]
            List of aggregation methods to apply.

        drop_original : bool, optional
            Whether to drop the original columns after transformation. Default is False.

        verbose: bool
            Controls  verbosity  of transformer

        """
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            verbose=verbose,
        )

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        >>> transformer  = AggregateColumnsOverRowTransformer(
        ... columns=['a', 'b'],
        ... aggregations=['min', 'max'],
        ... )

        >>> transformer.get_feature_names_out()
        ['a_b_min', 'a_b_max']

        """
        return ["_".join(self.columns) + "_" + agg for agg in self.aggregations]

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the dataframe by aggregating provided columns over each row.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating provided columns over each row

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Transformed DataFrame with aggregated columns.

        Example:
        --------
        >>> import polars as pl

        >>> transformer = AggregateColumnsOverRowTransformer(
        ... columns=['a', 'b'],
        ... aggregations=['min', 'max'],
        ... )

        >>> test_df = pl.DataFrame({'a': [1,2], 'b': [3,4], 'c':[5,6]})

        >>> transformer.transform(test_df)
        shape: (2, 5)
        ┌─────┬─────┬─────┬─────────┬─────────┐
        │ a   ┆ b   ┆ c   ┆ a_b_min ┆ a_b_max │
        │ --- ┆ --- ┆ --- ┆ ---     ┆ ---     │
        │ i64 ┆ i64 ┆ i64 ┆ i64     ┆ i64     │
        ╞═════╪═════╪═════╪═════════╪═════════╡
        │ 1   ┆ 3   ┆ 5   ┆ 1       ┆ 3       │
        │ 2   ┆ 4   ┆ 6   ┆ 2       ┆ 4       │
        └─────┴─────┴─────┴─────────┴─────────┘

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        expr_map = {
            "min": nw.min_horizontal(*self.columns),
            "max": nw.max_horizontal(*self.columns),
            "sum": nw.sum_horizontal(*self.columns),
            "mean": nw.mean_horizontal(*self.columns),
        }

        transform_dict = {
            "_".join(self.columns) + "_" + aggregation: expr_map[aggregation].alias(
                "_".join(self.columns) + "_" + aggregation,
            )
            for aggregation in self.aggregations
        }

        X = X.with_columns(**transform_dict)

        X = self.drop_original_column(
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )

        # Use mixin method to drop original columns
        return _return_narwhals_or_native_dataframe(X, self.return_native)
