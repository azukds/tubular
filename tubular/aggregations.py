from enum import Enum

import narwhals as nw
import narwhals.selectors as ncs
from beartype import beartype
from beartype.typing import Annotated, List
from beartype.vale import Is
from narwhals.typing import FrameT

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin


class AggregationOptions(str, Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"


ListOfAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in AggregationOptions._value2member_map_ for entry in list_value
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
        aggregations: ListOfAggregations,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(columns=columns, verbose=verbose)

        self.aggregations = aggregations

        self.drop_original = drop_original

    @beartype
    def create_new_col_names(self, prefix: str) -> list[str]:
        """Automatically generate new column names based on the aggregation type.

        Parameters
        ----------
        prefix : str
            Prefix to use for generating new column names.

        Returns
        -------
        list[str]
            List of new column names with the specified prefix and aggregation type.
        """
        return [f"{prefix}_{agg}" for agg in self.aggregations]

    @nw.narwhalify
    def transform(
        self,
        X: FrameT,
    ) -> FrameT:
        """Performs pre-transform safety checks.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating specified columns.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            checked dataframe to transform.

        Raises
        ------
        ValueError
            If columns are non-numeric.
        """

        super().transform(X)

        numerical_columns = X[self.columns].select(ncs.numeric()).columns

        non_numerical_columns = set(self.columns).difference(numerical_columns)

        # convert to list and sort for consistency in return
        non_numerical_columns = list(non_numerical_columns)
        non_numerical_columns.sort()
        if len(non_numerical_columns) != 0:
            msg = f"{self.classname}: attempting to call transformer on non-numeric columns {non_numerical_columns}, which is not supported"
            raise TypeError(msg)

        return X


class AggregateRowOverColumnsTransformer(BaseAggregationTransformer):
    """Transformer that aggregates rows over specified columns.

    This transformer aggregates data within specified columns, grouping by a key column,
    and can optionally drop the original columns post-transformation.

    Parameters
    ----------
    columns : list[str]
        List of column names to apply the aggregation transformations to.
    aggregations : list[str]
        List of aggregation methods to apply.
    key : str
        Column name to group by for aggregation.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.

    Attributes
    ----------
    key : str
        Column used for grouping during aggregation.
    """

    polars_compatible = True

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: ListOfAggregations,
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

    @nw.narwhalify
    def transform(
        self,
        X: FrameT,
    ) -> FrameT:
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

        super().transform(X)

        if self.key not in X.columns:
            msg = f"{self.classname()}: key '{self.key}' not found in dataframe columns"
            raise ValueError(msg)

        aggregation_dict = {col: self.aggregations for col in self.columns}

        grouped_X = X.group_by(self.key).agg(
            *(
                getattr(nw.col(col_name), agg)().alias(f"{col_name}_{agg}")
                for col_name, aggs in aggregation_dict.items()
                for agg in aggs
            ),
        )

        # Merge the aggregated results back with the original DataFrame
        X = X.join(grouped_X, on=self.key, how="left")

        # Use mixin method to drop original columns
        return self.drop_original_column(
            X,
            self.drop_original,
            self.columns,
        )
