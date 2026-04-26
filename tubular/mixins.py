"""Contains mixin classes for use across transformers."""

from __future__ import annotations

import warnings

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
        X: DataFrame
            Data containing columns to check.

        return_native: bool
            indicates whether to return nw or pd/pl dataframe

        Returns
        -------
        DataFrame:
            validated dataframe

        Raises
        ------
        TypeError:
            if provided columns are non-numeric

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
        return_native: bool = True,
        verbose: bool = False,
    ) -> tuple[DataFrame, str]:
        """Create unit weights column.

        Useful to streamline logic and just treat all cases as weighted,
        avoids branches for weights/non-weights.

        Function will check:
        - does 'unit_weights_column' already exist in data? (unlikely but
        check to be thorough)
        - if it does not, create unit weight 'unit_weights_column'
        - if it does, then reuse column
        - is it valid for our purposes? i.e. all unit weights
        - if not, raise warning (for verbose=True)

        Parameters
        ----------
        X: DataFrame
            pandas, polars, or narwhals df

        return_native: bool
            controls whether to return nw or pd/pl dataframe

        verbose:
            controls verbosity

        Returns
        -------
        DataFrame:
            DataFrame with added 'unit_weights_column'

        Raises
        ------
        TypeError: if unit_weights_column already exists and is non numeric.

        """
        X = _convert_dataframe_to_narwhals(X)

        unit_weights_column = "unit_weights_column"

        if unit_weights_column in X.columns:
            if X.schema[unit_weights_column] not in NumericTypes:
                error_msg = f"{unit_weights_column} is present in X and non-numeric, transformer logic requires this to be an all 1 value column."
                raise TypeError(
                    error_msg,
                )

            if verbose:
                warn_msg = f"column {unit_weights_column} is present in X, transformer logic will assume this column contains all 1 values."
                warnings.warn(warn_msg, stacklevel=2)

        else:
            # finally create dummy weights column if valid option not found
            X = X.with_columns(nw.lit(1).alias(unit_weights_column).cast(nw.Int8))

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
    def get_valid_weights_filter_expr(
        weights_column: str, verbose: bool = False
    ) -> nw.Expr:
        """Validate weights column in dataframe.

        Parameters
        ----------
        weights_column: str
            name of weight column
        verbose: bool
            control verbosity of method

        Returns
        -------
            nw.Expr: expression to be used for filtering down to valid weights rows

        """
        if verbose:
            warnings.warn(
                "Weights must be strictly positive, non-null, and finite - rows failing this will be filtered out.",
                stacklevel=2,
            )

        expr_ge_0 = nw.col(weights_column) > 0
        expr_not_null = ~nw.col(weights_column).is_null()
        expr_not_nan = ~nw.col(weights_column).is_nan()
        expr_finite = nw.col(weights_column).is_finite()

        return expr_ge_0 & expr_not_null & expr_not_nan & expr_finite
