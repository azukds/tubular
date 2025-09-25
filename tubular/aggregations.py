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
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    # not currently easy to implement row-wise
    # median or count, so leaving out for now


class RowsOverColumnsAggregationOptions(str, Enum):
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

    Attributes
    ----------
    columns : list[str]
        Columns to apply the transformations to.
    aggregations : list[str]
        Aggregation methods to apply.
    drop_original : bool
        Indicator for dropping original columns.
    verbose : bool
        Indicator for verbose output.
    """

    polars_compatible = True

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: Union[
            ListOfColumnsOverRowAggregations,
            ListOfRowsOverColumnsAggregations,
        ],
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(columns=columns, verbose=verbose)

        self.aggregations = aggregations

        self.drop_original = drop_original

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Performs pre-transform safety checks.

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
        ValueError
            If columns are non-numeric.
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
    """Transformer that aggregates rows over specified columns, where rows are grouped
    by provided key column.

    Attributes
    ----------
    columns : list[str]
        List of column names to apply the aggregation transformations to.
    aggregations : list[str]
        List of aggregation methods to apply.
    key : str
        Column name to group by for aggregation.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.
    """

    polars_compatible = True

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: ListOfRowsOverColumnsAggregations,
        key: str,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            verbose=verbose,
        )
        self.key = key

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transforms the dataframe by aggregating rows over specified columns.

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
    """Transformer that aggregates provided columns over each row

    This transformer aggregates data within specified columns
    and can optionally drop the original columns post-transformation.

    Attributes
    ----------
    columns : list[str]
        List of column names to apply the aggregation transformations to.
    aggregations : list[str]
        List of aggregation methods to apply.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.
    verbose : bool, optional
        Indicator for verbose output.
    """

    polars_compatible = True

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: ListOfColumnsOverRowAggregations,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            verbose=verbose,
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transforms the dataframe by aggregating provided columns over each row

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating provided columns over each row

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Transformed DataFrame with aggregated columns.

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
