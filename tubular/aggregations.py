import narwhals as nw
from beartype import beartype
from beartype.typing import List, Literal
from narwhals import count, mean, median, mode  # Example import statement
from narwhals.typing import FrameT

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin


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
        'mean', 'median', 'mode', 'sum', and 'count'.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.
    level : str, optional
        Specifies the level of aggregation, either 'row' or 'column'. Default is 'row'.

    Attributes
    ----------
    columns : list[str]
        Columns to apply the transformations to.
    aggregations : list[str]
        Aggregation methods to apply.
    drop_original : bool
        Indicator for dropping original columns.
    level : str
        Level of aggregation.
    """

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: List[
            Literal["min", "max", "mean", "median", "mode", "sum", "count"]
        ],
        drop_original: bool = False,
        level: Literal["row", "column"] = "row",
    ) -> None:
        self.columns = columns
        self.aggregations = aggregations
        self.drop_original = drop_original
        self.level = level

        self.set_drop_original_column(drop_original)

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

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: list[str],
        key: str,
        drop_original: bool = False,
    ) -> None:
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            level="row",
        )
        self.key = key

    @nw.narwhalify
    def transform(
        self,
        df: FrameT,
    ) -> FrameT:
        """Transforms the dataframe by aggregating rows over specified columns.

        Parameters
        ----------
        df : FrameT
            DataFrame to transform by aggregating specified columns.

        Returns
        -------
        FrameT
            Transformed DataFrame with aggregated columns.

        Raises
        ------
        ValueError
            If the key column is not found in the DataFrame.
        """
        if self.key not in df.columns:
            msg = f"key '{self.key}' not found in dataframe columns"
            raise ValueError(msg)

        aggregation_dict = {col: self.aggregations for col in self.columns}
        grouped_df = df.group_by(self.key).agg(
            *(
                getattr(nw.col(col_name), agg)().alias(f"{col_name}_{agg}")
                for col_name, aggs in aggregation_dict.items()
                for agg in aggs
            ),
        )

        # Merge the aggregated results back with the original DataFrame
        df = df.join(grouped_df, on=self.key, how="left")

        # Use mixin method to drop original columns
        return self.drop_original_column(
            df,
            self.drop_original,
            self.columns,
        )
