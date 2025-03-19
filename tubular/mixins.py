from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import narwhals.selectors as ncs
import numpy as np

if TYPE_CHECKING:
    from narhwals.typing import FrameT


class CheckNumericMixin:
    """
    Mixin class with methods for numeric transformers

    """

    def classname(self) -> str:
        """Method that returns the name of the current class when called."""

        return type(self).__name__

    @nw.narwhalify
    def check_numeric_columns(self, X: FrameT) -> FrameT:
        """Helper function for checking column args are numeric for numeric transformers.

        Args:
        ----
            X: Data containing columns to check.

        """
        non_numeric_columns = list(
            set(self.columns).difference(set(X.select(ncs.numeric()).columns)),
        )
        # sort as set ordering can be inconsistent
        non_numeric_columns.sort()
        if len(non_numeric_columns) > 0:
            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        return X


class DropOriginalMixin:
    """Mixin class to validate and apply 'drop_original' argument used by various transformers.

    Transformer deletes transformer input columns depending on boolean argument.

    """

    def classname(self) -> str:
        """Method that returns the name of the current class when called."""

        return type(self).__name__

    def set_drop_original_column(self, drop_original: bool) -> None:
        """Helper method for validating 'drop_original' argument.

        Parameters
        ----------
        drop_original : bool
            boolean dictating dropping the input columns from X after checks.

        """
        # check if 'drop_original' argument is boolean
        if type(drop_original) is not bool:
            msg = f"{self.classname()}: drop_original should be bool"
            raise TypeError(msg)

        self.drop_original = drop_original

    @nw.narwhalify
    def drop_original_column(
        self,
        X: FrameT,
        drop_original: bool,
        columns: list[str] | str | None,
    ) -> FrameT:
        """Method for dropping input columns from X if drop_original set to True.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data with columns to drop.

        drop_original : bool
            boolean dictating dropping the input columns from X after checks.

        columns: list[str] | str |  None
            Object containing columns to drop

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with columns dropped.

        """

        if drop_original:
            X = X.drop(columns)

        return X


class NewColumnNameMixin:
    """Helper to validate and set new_column_name attribute"""

    def check_and_set_new_column_name(self, new_column_name: str) -> None:
        if not (isinstance(new_column_name, str)):
            msg = f"{self.classname()}: new_column_name should be str"
            raise TypeError(msg)

        self.new_column_name = new_column_name


class SeparatorColumnMixin:
    """Hel per to validate and set separator attribute"""

    def check_and_set_separator_column(self, separator: str) -> None:
        if not (isinstance(separator, str)):
            msg = f"{self.classname()}: separator should be str"
            raise TypeError(msg)

        self.separator = separator


class TwoColumnMixin:
    """helper to validate columns when exactly two columns are required"""

    def check_two_columns(self, columns: list[str]) -> None:
        if not (isinstance(columns, list)):
            msg = f"{self.classname()}: columns should be list"
            raise TypeError(msg)

        if len(columns) != 2:
            msg = f"{self.classname()}: This transformer works with two columns only"
            raise ValueError(msg)


class WeightColumnMixin:
    """
    Mixin class with weights functionality

    """

    def classname(self) -> str:
        """Method that returns the name of the current class when called."""
        return type(self).__name__

    @nw.narwhalify
    def check_weights_column(self, X: FrameT, weights_column: str) -> None:
        """Helper method for validating weights column in dataframe.

        Args:
        ----
            X: pandas or polars df containing weight column
            weights_column: name of weight column

        """
        # check if given weight is in columns
        if weights_column not in X.columns:
            msg = f"{self.classname()}: weight col ({weights_column}) is not present in columns of data"
            raise ValueError(msg)

        # check weight is numeric
        if weights_column not in X.select(ncs.numeric()).columns:
            msg = f"{self.classname()}: weight column must be numeric."
            raise ValueError(msg)

        # check weight is positive
        if X.select(nw.col(weights_column).min()).item() < 0:
            msg = f"{self.classname()}: weight column must be positive"
            raise ValueError(msg)

        if (
            # check weight non-None
            (X.select(nw.col(weights_column).is_null().sum()).item() != 0)
            or
            # TODO - update this section with narwhals is_nan method once this becomes
            # available
            # check weight non-NaN - polars differentiates between None and NaN
            (np.isnan(X.get_column(weights_column).to_numpy()).sum() != 0)
        ):
            msg = f"{self.classname()}: weight column must be non-null"
            raise ValueError(msg)

        # check weight not inf
        if not X.select((nw.col(weights_column).is_finite()).all()).item():
            msg = f"{self.classname()}: weight column must not contain infinite values."
            raise ValueError(msg)

        # check weight not all 0
        if X.select(nw.col(weights_column).sum()).item() == 0:
            msg = f"{self.classname()}: total sample weights are not greater than 0"
            raise ValueError(msg)

    def check_and_set_weight(self, weights_column: str) -> None:
        """Helper method that validates and assigns the specified column name to be used as the weights_column attribute.
        This function ensures that the `weights_column` parameter is either a string representing
        the column name or None. If `weights_column` is not of type str and is not None, it raises
        a TypeError.

        Parameters:
            weights_column (str or None): The name of the column to be used as weights. If None, no weights are used.

        Raises:
            TypeError: If `weights_column` is neither a string nor None.

        Returns:
            None
        """

        if weights_column is not None and not isinstance(weights_column, str):
            msg = "weights_column should be str or None"
            raise TypeError(msg)
        self.weights_column = weights_column
