"""Contains mixin classes for use across transformers."""

from __future__ import annotations

from typing import Literal, Optional, Union

import narwhals as nw
from beartype import beartype
from narwhals.dtypes import DType  # noqa: F401 - required for nw.Schema see #455

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
)
from tubular.types import DataFrame, NumericTypes


class CheckNumericMixin:
    """Mixin class with methods for numeric transformers."""

    def classname(self) -> str:
        """Get name of the current class when called.

        Returns
        -------
            str:
                name of class

        """
        return type(self).__name__

    @beartype
    def check_numeric_columns(
        self,
        X: DataFrame,
        return_native: bool = True,
    ) -> DataFrame:
        """Check column args are numeric for numeric transformers.

        Parameters
        ----------
        X: pd/pl/nw.DataFrame
            Data containing columns to check.

        return_native: bool
            indicates whether to return nw or pd/pl dataframe

        Raises
        ------
            TypeError:
                if provided columns are non-numeric

        Returns
        -------
            pd/pl/nw.DataFrame:
                validated dataframe

        """
        X = _convert_dataframe_to_narwhals(X)
        schema = X.schema

        non_numeric_columns = [
            col for col in self.columns if schema[col] not in NumericTypes
        ]

        # sort as set ordering can be inconsistent
        non_numeric_columns.sort()
        if len(non_numeric_columns) > 0:
            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        return _return_narwhals_or_native_dataframe(X, return_native)


class DropOriginalMixin:
    """Mixin class to validate and apply 'drop_original' argument used by various transformers.

    Transformer deletes transformer input columns depending on boolean argument.

    """

    def classname(self) -> str:
        """Get name of the current class when called.

        Returns
        -------
            str:
                name of class

        """
        return type(self).__name__

    @staticmethod
    @beartype
    @nw.narwhalify
    def drop_original_column(
        X: DataFrame,
        drop_original: bool,
        columns: Optional[Union[list[str], str]],
        return_native: bool = True,
    ) -> DataFrame:
        """Drop input columns from X if drop_original set to True.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data with columns to drop.

        drop_original : bool
            boolean dictating dropping the input columns from X after checks.

        columns: list[str] | str |  None
            Object containing columns to drop

        return_native: bool
            controls whether mixin returns native or narwhals type

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with columns dropped.

        """
        X = _convert_dataframe_to_narwhals(X)

        if drop_original:
            X = X.drop(columns)

        return X.to_native() if return_native else X


class WeightColumnMixin:
    """Mixin class with weights functionality."""

    def classname(self) -> str:
        """Get the name of the current class when called.

        Returns
        -------
            str:
                name of class

        """
        return type(self).__name__

    @staticmethod
    def _create_unit_weights_column(
        X: DataFrame,
        backend: Literal["pandas", "polars"],
        return_native: bool = True,
    ) -> tuple[DataFrame, str]:
        """Create unit weights column.

        Useful to streamline logic and just treat all cases as weighted,
        avoids branches for weights/non-weights.

        Function will check:
        - does 'unit_weights_column' already exist in data? (unlikely but
        check to be thorough)
        - if it does not, create unit weight 'unit_weights_column'
        - if it does, is it valid for our purposes? i.e. all unit weights
        - if it is, then just reuse this existing column
        - if is not, throw error

        Args:
        ----
            X: DataFrame
                pandas, polars, or narwhals df

            backend: Literal['pandas', 'polars']
                backed of original df

            return_native: bool
                controls whether to return nw or pd/pl dataframe

        Raises:
        ------
            RuntimeError:
                if invalid 'unit_weights_column' already exists

        Returns:
            pd/pl/nw.DataFrame:
                DataFrame with added 'unit_weights_column'

        """
        X = _convert_dataframe_to_narwhals(X)

        unit_weights_column = "unit_weights_column"

        if unit_weights_column in X.columns:
            all_one = len(X.filter(nw.col(unit_weights_column) == 1)) == len(
                X,
            )
            # if exists already and is valid, return
            if all_one:
                return _return_narwhals_or_native_dataframe(
                    X,
                    return_native,
                ), unit_weights_column

            # error if column already exists but is not suitable
            msg = "Attempting to insert column of unit weights named 'unit_weights_column', but an existing column shares this name and is not all 1, please rename existing column"
            raise RuntimeError(
                msg,
            )

        # finally create dummy weights column if valid option not found
        X = X.with_columns(
            nw.new_series(
                name=unit_weights_column,
                values=[1] * len(X),
                backend=backend,
            ),
        )

        return _return_narwhals_or_native_dataframe(
            X,
            return_native,
        ), unit_weights_column

    @beartype
    def check_weights_column(self, X: DataFrame, weights_column: str) -> None:
        """Validate weights column in dataframe.

        Parameters
        ----------
        X: DataFrame
            input data
        weights_column: str
            name of weight column

        Raises
        ------
            ValueError:
                if weights_column is missing from data

            ValueError:
                if weights_column is non-numeric

        """
        X = _convert_dataframe_to_narwhals(X)

        # check if given weight is in columns
        if weights_column not in X.columns:
            msg = f"{self.classname()}: weight col ({weights_column}) is not present in columns of data"
            raise ValueError(msg)

        # check weight is numeric
        schema = X.schema
        if schema[weights_column] not in NumericTypes:
            msg = f"{self.classname()}: weight column must be numeric."
            raise ValueError(msg)

    @staticmethod
    @beartype
    def get_valid_weights_filter_expr(weights_column: str) -> nw.Expr:
        """Validate weights column in dataframe.

        Parameters
        ----------
        weights_column: str
            name of weight column

        Returns
        -------
            nw.Expr: expression to be used for filtering down to valid weights rows

        """
        expr_ge_0 = nw.col(weights_column) > 0
        expr_not_null = ~nw.col(weights_column).is_null()
        expr_not_nan = ~nw.col(weights_column).is_nan()
        expr_finite = nw.col(weights_column).is_finite()

        return expr_ge_0 & expr_not_null & expr_not_nan & expr_finite
