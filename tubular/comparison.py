"""Contains transformers for performing data comparisons."""

from __future__ import annotations

from beartype import beartype
from typing_extensions import deprecated

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin
from tubular.types import ListOfTwoStrs


@deprecated(
    "EqualityChecker is deprecated and will be removed in a future version. "
    "Please use a different approach for your use case. If you have a specific "
    "use case, please open an issue for it to be modernised",
)
class EqualityChecker(
    DropOriginalMixin,
    BaseTransformer,
):
    """Check if two columns are equal.

    This transformer creates a new column that contains boolean values indicating
    whether the values in two specified columns are equal.

    Parameters
    ----------
    columns : list of two str
        Two column names to compare for equality.
    new_column_name : str
        Name for the new column containing the equality check results.
    drop_original : bool, default=False
        Whether to drop the original columns after creating the new column.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    columns : list of two str
        Two column names to compare for equality.
    new_column_name : str
        Name for the new column containing the equality check results.
    drop_original : bool
        Whether to drop the original columns after creating the new column.

    .. deprecated:: 0.5.0
        This transformer is deprecated and will be removed in a future version.

    Examples
    --------
    >>> from tubular.comparison import EqualityChecker
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 4]})
    >>> transformer = EqualityChecker(columns=['a', 'b'], new_column_name='equal')
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed[['a', 'b', 'equal']])
       a  b  equal
    0  1  1   True
    1  2  2   True
    2  3  4  False

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = True

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        new_column_name: str,
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize EqualityChecker.

        Parameters
        ----------
        columns : list of two str
            Two column names to compare for equality.
        new_column_name : str
            Name for the new column containing the equality check results.
        drop_original : bool, default=False
            Whether to drop the original columns after creating the new column.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        super().__init__(columns=columns, copy=copy, verbose=verbose, return_native=return_native)
        self.new_column_name = new_column_name
        self.drop_original = drop_original

    @beartype
    def transform(
        self,
        X,
    ):
        """Transform the dataframe by checking column equality.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.

        Returns
        -------
        narwhals Frame
            Transformed dataframe with equality check column added.

        """
        import narwhals as nw

        X = super().transform(X, return_native_override=False)

        # Check equality
        X = X.with_columns(
            (X[self.columns[0]] == X[self.columns[1]]).alias(self.new_column_name)
        )

        # Drop original columns if requested
        if self.drop_original:
            X = X.drop(self.columns)

        from tubular._utils import _return_narwhals_or_native_dataframe

        return _return_narwhals_or_native_dataframe(X, self.return_native)
