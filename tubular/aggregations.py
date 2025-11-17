"""Contains transformers for performing data aggregations."""

from enum import Enum
from typing import Any, Union

import narwhals as nw
from beartype import beartype
from beartype.typing import Annotated, List, Optional
from beartype.vale import Is

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin


class RowsOverColumnsAggregationOptions(str, Enum):
    """Enumeration of valid aggregation options for rows over columns."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    COUNT = "count"
    NUNIQUE = "nunique"


class ColumnsOverRowsAggregationOptions(str, Enum):
    """Enumeration of valid aggregation options for columns over rows."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"


class BaseAggregationTransformer(BaseTransformer, DropOriginalMixin):
    """Base class for aggregation transformers.

    This class provides common functionality for transformers that perform
    aggregations on dataframes. It inherits from both BaseTransformer and
    DropOriginalMixin to provide standard transformer behavior and the ability
    to drop original columns.

    Parameters
    ----------
    columns : str or list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply. Valid options depend on the
        specific transformer subclass.
    drop_original : bool, default=False
        Whether to drop the original columns after creating aggregated columns.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    columns : list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply.
    drop_original : bool
        Whether to drop the original columns after creating aggregated columns.

    Notes
    -----
    This is a base class and should not be instantiated directly. Use one of
    the subclasses instead.

    Examples
    --------
    >>> from tubular.aggregations import BaseAggregationTransformer
    >>> # This will raise an error as BaseAggregationTransformer is abstract
    >>> transformer = BaseAggregationTransformer(columns='a', aggregations=['min'])
    Traceback (most recent call last):
        ...
    NotImplementedError: BaseAggregationTransformer is an abstract class

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = True

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[str, List[str]],
        aggregations: List[str],
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize BaseAggregationTransformer.

        Parameters
        ----------
        columns : str or list of str
            Columns to apply aggregations to.
        aggregations : list of str
            List of aggregation functions to apply.
        drop_original : bool, default=False
            Whether to drop the original columns after creating aggregated columns.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        if self.__class__ == BaseAggregationTransformer:
            raise NotImplementedError("BaseAggregationTransformer is an abstract class")

        super().__init__(columns=columns, copy=copy, verbose=verbose, return_native=return_native)

        self.aggregations = aggregations
        self.drop_original = drop_original

    @beartype
    def transform(
        self,
        X: Annotated[
            nw.TypedFrame,
            Is[
                lambda frame: nw.is_dataframe(frame)
                and nw.len(frame) > 0
                and nw.n_cols(frame) > 0
            ],
        ],
    ) -> nw.TypedFrame:
        """Transform the dataframe by applying aggregations.

        This method should be overridden by subclasses to implement specific
        aggregation logic.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.

        Returns
        -------
        narwhals Frame
            Transformed dataframe with aggregated columns.

        Raises
        ------
        NotImplementedError
            If called on BaseAggregationTransformer directly.

        """
        raise NotImplementedError("Subclasses must implement transform method")

        return _return_narwhals_or_native_dataframe(X, return_native=return_native)

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        >>> transformer = BaseAggregationTransformer(
        ...     columns='a',
        ...     aggregations=['min', 'max'],
        ... )
        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseAggregationTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'aggregations': ['min', 'max'], 'drop_original': False}, 'fit': {}}

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "aggregations": self.aggregations,
                "drop_original": self.drop_original,
            }
        )

        return json_dict


class AggregateRowsOverColumnTransformer(BaseAggregationTransformer):
    """Aggregate rows over specified columns, where rows are grouped by provided key column.

    This transformer groups rows by a key column and then applies aggregation
    functions to specified columns within each group. The aggregated values are
    then merged back to the original dataframe.

    Parameters
    ----------
    columns : str or list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply. Valid options are:
        - 'sum': Sum of values
        - 'mean': Mean of values
        - 'median': Median of values
        - 'min': Minimum value
        - 'max': Maximum value
        - 'std': Standard deviation
        - 'var': Variance
        - 'count': Count of non-null values
        - 'nunique': Number of unique values
    key : str
        Column name to group rows by before applying aggregations.
    drop_original : bool, default=False
        Whether to drop the original columns after creating aggregated columns.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    columns : list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply.
    key : str
        Column name to group rows by.
    drop_original : bool
        Whether to drop the original columns after creating aggregated columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from tubular.aggregations import AggregateRowsOverColumnTransformer
    >>> df = pd.DataFrame({
    ...     'group': ['A', 'A', 'B', 'B'],
    ...     'value': [1, 2, 3, 4],
    ...     'other': [10, 20, 30, 40]
    ... })
    >>> transformer = AggregateRowsOverColumnTransformer(
    ...     columns='value',
    ...     aggregations=['mean', 'sum'],
    ...     key='group'
    ... )
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       group  value  other  value_mean  value_sum
    0      A      1     10         1.5          3
    1      A      2     20         1.5          3
    2      B      3     30         3.5          7
    3      B      4     40         3.5          7

    Notes
    -----
    The transformer creates new columns with names in the format
    '{column}_{aggregation}' for each combination of column and aggregation.

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = True

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[str, List[str]],
        aggregations: List[str],
        key: str,
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize AggregateRowsOverColumnTransformer.

        Parameters
        ----------
        columns : str or list of str
            Columns to apply aggregations to.
        aggregations : list of str
            List of aggregation functions to apply.
        key : str
            Column name to group rows by.
        drop_original : bool, default=False
            Whether to drop the original columns after creating aggregated columns.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            copy=copy,
            verbose=verbose,
            return_native=return_native,
        )
        self.key = key

    @block_from_json
    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list of str
            List of feature names created by the transformer. Format is
            '{column}_{aggregation}' for each combination of column and aggregation.

        Examples
        --------
        >>> transformer = AggregateRowsOverColumnTransformer(
        ...     columns=['a', 'b'],
        ...     aggregations=['min', 'max'],
        ...     key='group'
        ... )
        >>> transformer.get_feature_names_out()
        ['a_min', 'a_max', 'b_min', 'b_max']

        """
        return [f"{col}_{agg}" for col in self.columns for agg in self.aggregations]

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        >>> transformer = AggregateRowsOverColumnTransformer(
        ...     columns='a',
        ...     aggregations=['min', 'max'],
        ...     key='b',
        ... )
        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'AggregateRowsOverColumnTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'aggregations': ['min', 'max'], 'drop_original': False, 'key': 'b'}, 'fit': {}}

        """
        json_dict = super().to_json()

        json_dict["init"]["key"] = self.key

        return json_dict

    @beartype
    def transform(
        self,
        X: Annotated[
            nw.TypedFrame,
            Is[
                lambda frame: nw.is_dataframe(frame)
                and nw.len(frame) > 0
                and nw.n_cols(frame) > 0
            ],
        ],
    ) -> nw.TypedFrame:
        """Transform the dataframe by aggregating rows over columns.

        Groups rows by the key column and applies aggregation functions to
        the specified columns. The aggregated values are merged back to the
        original dataframe.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.

        Returns
        -------
        narwhals Frame
            Transformed dataframe with aggregated columns added.

        Examples
        --------
        >>> import pandas as pd
        >>> from tubular.aggregations import AggregateRowsOverColumnTransformer
        >>> df = pd.DataFrame({
        ...     'group': ['A', 'A', 'B', 'B'],
        ...     'value': [1, 2, 3, 4]
        ... })
        >>> transformer = AggregateRowsOverColumnTransformer(
        ...     columns='value',
        ...     aggregations=['mean'],
        ...     key='group'
        ... )
        >>> df_transformed = transformer.transform(df)
        >>> print(df_transformed[['group', 'value', 'value_mean']])
           group  value  value_mean
        0      A      1         1.5
        1      A      2         1.5
        2      B      3         3.5
        3      B      4         3.5

        """
        X = _convert_dataframe_to_narwhals(X)

        # Validate aggregations
        valid_aggregations = [opt.value for opt in RowsOverColumnsAggregationOptions]
        invalid_aggregations = [agg for agg in self.aggregations if agg not in valid_aggregations]
        if invalid_aggregations:
            raise ValueError(
                f"Invalid aggregation(s): {invalid_aggregations}. "
                f"Valid options are: {valid_aggregations}"
            )

        # Group by key and aggregate
        grouped = X.group_by(self.key)
        agg_expressions = []
        for col in self.columns:
            for agg in self.aggregations:
                agg_func = getattr(nw, agg)
                agg_col_name = f"{col}_{agg}"
                agg_expressions.append(agg_func(grouped[col]).alias(agg_col_name))

        aggregated = grouped.agg(agg_expressions)

        # Merge aggregated results back to original dataframe
        X = X.join(aggregated, on=self.key, how="left")

        # Drop original columns if requested
        if self.drop_original:
            X = X.drop(self.columns)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class AggregateColumnsOverRowTransformer(BaseAggregationTransformer):
    """Aggregate provided columns over each row.

    This transformer applies aggregation functions across specified columns
    for each row, creating new columns with the aggregated values.

    Parameters
    ----------
    columns : str or list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply. Valid options are:
        - 'sum': Sum of values
        - 'mean': Mean of values
        - 'median': Median of values
        - 'min': Minimum value
        - 'max': Maximum value
        - 'std': Standard deviation
        - 'var': Variance
    drop_original : bool, default=False
        Whether to drop the original columns after creating aggregated columns.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    columns : list of str
        Columns to apply aggregations to.
    aggregations : list of str
        List of aggregation functions to apply.
    drop_original : bool
        Whether to drop the original columns after creating aggregated columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from tubular.aggregations import AggregateColumnsOverRowTransformer
    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [4, 5, 6],
    ...     'c': [7, 8, 9]
    ... })
    >>> transformer = AggregateColumnsOverRowTransformer(
    ...     columns=['a', 'b', 'c'],
    ...     aggregations=['mean', 'sum']
    ... )
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed[['a', 'b', 'c', 'row_mean', 'row_sum']])
       a  b  c  row_mean  row_sum
    0  1  4  7       4.0       12
    1  2  5  8       5.0       15
    2  3  6  9       6.0       18

    Notes
    -----
    The transformer creates new columns with names in the format
    'row_{aggregation}' for each aggregation function.

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = True

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Union[str, List[str]],
        aggregations: List[str],
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize AggregateColumnsOverRowTransformer.

        Parameters
        ----------
        columns : str or list of str
            Columns to apply aggregations to.
        aggregations : list of str
            List of aggregation functions to apply.
        drop_original : bool, default=False
            Whether to drop the original columns after creating aggregated columns.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            copy=copy,
            verbose=verbose,
            return_native=return_native,
        )

    @beartype
    def transform(
        self,
        X: Annotated[
            nw.TypedFrame,
            Is[
                lambda frame: nw.is_dataframe(frame)
                and nw.len(frame) > 0
                and nw.n_cols(frame) > 0
            ],
        ],
    ) -> nw.TypedFrame:
        """Transform the dataframe by aggregating columns over rows.

        Applies aggregation functions across specified columns for each row,
        creating new columns with the aggregated values.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.

        Returns
        -------
        narwhals Frame
            Transformed dataframe with aggregated columns added.

        Examples
        --------
        >>> import pandas as pd
        >>> from tubular.aggregations import AggregateColumnsOverRowTransformer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2],
        ...     'b': [3, 4]
        ... })
        >>> transformer = AggregateColumnsOverRowTransformer(
        ...     columns=['a', 'b'],
        ...     aggregations=['mean']
        ... )
        >>> df_transformed = transformer.transform(df)
        >>> print(df_transformed[['a', 'b', 'row_mean']])
           a  b  row_mean
        0  1  3       2.0
        1  2  4       3.0

        """
        X = _convert_dataframe_to_narwhals(X)

        # Validate aggregations
        valid_aggregations = [opt.value for opt in ColumnsOverRowsAggregationOptions]
        invalid_aggregations = [agg for agg in self.aggregations if agg not in valid_aggregations]
        if invalid_aggregations:
            raise ValueError(
                f"Invalid aggregation(s): {invalid_aggregations}. "
                f"Valid options are: {valid_aggregations}"
            )

        # Create aggregation expressions
        agg_expressions = []
        for agg in self.aggregations:
            agg_func = getattr(nw, agg)
            agg_col_name = f"row_{agg}"
            # Aggregate across columns for each row
            agg_expressions.append(
                agg_func([X[col] for col in self.columns]).alias(agg_col_name)
            )

        X = X.with_columns(agg_expressions)

        # Drop original columns if requested
        if self.drop_original:
            X = X.drop(self.columns)

        return _return_narwhals_or_native_dataframe(X, self.return_native)
