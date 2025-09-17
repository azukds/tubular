"""This module contains transformers that deal with imputation of missing values."""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Union

import narwhals as nw
import polars as pl
from beartype import beartype

from tubular._utils import (
    _assess_pandas_object_column,
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _return_narwhals_or_native_dataframe,
)
from tubular.base import BaseTransformer
from tubular.mixins import WeightColumnMixin
from tubular.types import DataFrame, Series

pl.enable_string_cache()


class BaseImputer(BaseTransformer):
    """Base imputer class containing standard transform method that will use pd.Series.fillna with the
    values in the impute_values_ attribute.

    Other imputers in this module should inherit from this class.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    FITS = False

    def _generate_imputation_expressions(self, expr: nw.Expr, col: str) -> nw.Expr:
        """update input expressions to include imputation.

        Parameters
        ----------
        expr : nw.Expr
            initial expression
        col: str
            column being imputed

        Returns
        -------
        nw.Expr: updated expression, with imputation

        """

        return (
            expr.fill_null(value=self.impute_values_[col])
            if (self.impute_values_[col] is not None)
            else expr
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Impute missing values with values calculated from fit method.

        Parameters
        ----------
        X : FrameT
            Data to impute.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : FrameT
            Transformed input X with nulls imputed with the median value for the specified columns.

        """
        self.check_is_fitted("impute_values_")

        return_native = self._process_return_native(return_native_override)

        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        transform_expressions = {
            col: self._generate_imputation_expressions(nw.col(col), col)
            for col in self.columns
        }

        X = X.with_columns(**transform_expressions)

        return _return_narwhals_or_native_dataframe(X, return_native)


class ArbitraryImputer(BaseImputer):
    """Transformer to impute null values with an arbitrary pre-defined value.

    Parameters
    ----------
    impute_value : int or float or str or bool
        Value to impute nulls with.
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.
    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_value : int or float or str or bool
        Value to impute nulls with.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type
    """

    polars_compatible = True
    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: Union[int, float, str, bool],
        columns: Union[str, list[str]],
        **kwargs: Optional[bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

    def cat_to_enum_expr(self, expr: nw.Expr, categories: list[str]) -> nw.Expr:
        """update expression to include handling of category types to allow new
        impute value category

        Parameters
        ----------
        expr : nw.Expr
            initial expression
        categories: list[str]
            list of categories in field initially

        Returns
        -------
        nw.Expr: updated expression, with category type handling

        """

        return expr.cast(nw.Enum(set(categories + [self.impute_value])))

    def _check_impute_value_type_works_with_columns(
        self,
        X: DataFrame,
        schema: nw.Schema,
        native_namespace: Literal["pandas", "polars"],
    ) -> tuple[dict[str, str], list[StopIteration]]:
        """raises TypeError if there is a type clash between impute_value and columns in X for imputation

        Parameters
        ----------
        X: FrameT
            DataFrame being imputed

        Returns
        ---------
        pandas_object_cols_to_polars_types: dict[str, str]
            dictionary of type conversions for tricky pandas object types

        null_columns: list[str]
            list of Unknown type columns, singled out for different type handling

        """

        object_columns = set()
        cat_columns = set()
        num_columns = set()
        bool_columns = set()
        str_columns = set()
        null_columns = set()
        for col in self.columns:
            dtype = schema[col]
            if dtype == nw.Object:
                object_columns.add(col)
            elif dtype == nw.Categorical:
                cat_columns.add(col)
            elif dtype in [
                nw.Float32,
                nw.Float64,
                nw.Int64,
                nw.Int32,
                nw.Int16,
                nw.Int8,
            ]:
                num_columns.add(col)
            elif dtype == nw.Boolean:
                bool_columns.add(col)
            elif dtype == nw.String:
                str_columns.add(col)
            elif dtype == nw.Unknown:
                null_columns.add(col)

        if len(cat_columns) > 0 and native_namespace == "pandas":
            warnings.warn(
                f"{self.classname()}: this transformer will convert unordered categorical columns to ordered for pandas dfs",
                stacklevel=2,
            )

        # start with object columns, which can be a massive nuisance from pandas
        pandas_object_cols_to_polars_types = {}
        if len(object_columns) > 0 and native_namespace == "pandas":
            # pull out boolean columns from generic object columns
            for col in object_columns:
                _, polars_type = _assess_pandas_object_column(
                    pandas_df=X.to_native(),
                    col=col,
                )
                pandas_object_cols_to_polars_types[col] = getattr(nw, polars_type)

                if polars_type == "Boolean":
                    bool_columns = bool_columns.union({col})

                # other types will be captured in error at end of this method

        if (not isinstance(self.impute_value, str)) and (
            len(cat_columns) > 0 or len(str_columns) > 0
        ):
            msg = f"""
                {self.classname()}: Attempting to impute non-str value {self.impute_value} into
                Categorical or String type columns, this is not type safe,
                please use str impute_value for these columns
                (this may require separate ArbitraryImputer instances for different column types)
                """
            raise TypeError(
                msg,
            )

        if (not isinstance(self.impute_value, (float, int))) and len(num_columns) > 0:
            msg = f"""
                {self.classname()}: Attempting to impute non-numeric value {self.impute_value} into
                Numeric type columns, this is not type safe,
                please use numeric impute_value for these columns
                (this may require separate ArbitraryImputer instances for different column types)
                """
            raise TypeError(
                msg,
            )

        if (not isinstance(self.impute_value, bool)) and len(bool_columns) > 0:
            msg = f"""
                {self.classname()}: Attempting to impute non-bool value {self.impute_value} into
                Boolean type columns, this is not type safe,
                please use bool impute_value for these columns
                (this may require separate ArbitraryImputer instances for different column types)
                """
            raise TypeError(
                msg,
            )

        if len(null_columns) > 0:
            warnings.warn(
                f"{self.classname()}: X contains all null columns {null_columns}, types for these columns will be inferred as {type(self.impute_value)}",
                stacklevel=2,
            )

        bad_type_cols = set(self.columns).difference(
            num_columns.union(bool_columns)
            .union(str_columns)
            .union(cat_columns)
            .union(null_columns),
        )
        if len(bad_type_cols) != 0:
            bad_types = {
                name: dtype for name, dtype in schema.items() if name in bad_type_cols
            }
            msg = f"""
                {self.classname()}: transformer can only handle Float/Int/Boolean/String/Categorical/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        return pandas_object_cols_to_polars_types, null_columns

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.
        If columns is None all columns in X will be imputed.

        Parameters
        ----------
        X : FrameT
            Data containing columns to impute.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : FrameT
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.
        """

        X = _convert_dataframe_to_narwhals(X)

        schema = X.schema
        native_namespace = nw.get_native_namespace(X).__name__

        X = BaseTransformer.transform(self, X, return_native_override=False)

        (
            pandas_object_cols_to_polars_types,
            null_columns,
        ) = self._check_impute_value_type_works_with_columns(
            X,
            schema,
            native_namespace,
        )

        # Save the original dtypes BEFORE we cast anything
        original_dtypes = {}
        for col in self.columns:
            original_dtypes[col] = (
                # overwrite type if necessary, e.g. object->boolean
                pandas_object_cols_to_polars_types[col]
                if col in pandas_object_cols_to_polars_types
                else schema[col]
            )

        # have to handle categorical vars for pandas upfront
        if native_namespace == "pandas":
            transform_expressions = {
                col: self.cat_to_enum_expr(
                    nw.col(col),
                    categories=X.get_column(col).cat.get_categories().to_list(),
                )
                if ((schema[col] == nw.Categorical) or (schema[col] == nw.Enum))
                else nw.col(col)
                for col in self.columns
            }
        else:
            transform_expressions = {col: nw.col(col) for col in self.columns}

        # next handle imputing
        transform_expressions = {
            col: self._generate_imputation_expressions(
                transform_expressions[col],
                col,
            )
            for col in self.columns
        }

        # finally manage types
        transform_expressions = {
            col: transform_expressions[col].cast(original_dtypes[col])
            if (col not in null_columns)
            else transform_expressions[col]
            for col in self.columns
        }

        X = X.with_columns(**transform_expressions)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class MedianImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the median of the supplied columns.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weights_column: None or str, default=None
        Column containing weights

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (median) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @beartype
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> MedianImputer:
        """Calculate median values to impute with from X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to "learn" the median values from.

        y : None or pd/pl.Series, default = None
            Not required.

        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        for c in self.columns:
            # filter out null rows so their weight doesn't influence calc
            filtered = X.filter(~nw.col(c).is_null())

            # if column is only nulls, then median is None
            if len(filtered) <= 0:
                self.impute_values_[c] = None

            elif self.weights_column is not None:
                WeightColumnMixin.check_weights_column(self, X, self.weights_column)

                # first sort df by column to be imputed (order of weight column shouldn't matter for median)
                filtered = filtered.sort(c)

                # next calculate cumulative weight sums
                cumsum = filtered[self.weights_column].cum_sum()

                # find midpoint
                cutoff = filtered[self.weights_column].sum() / 2.0

                # find first value >= this point
                median = filtered.filter(cumsum >= cutoff).select(c)[0].item()

                # impute value is weighted median
                self.impute_values_[c] = median

            else:
                # impute value is median without considering weight
                self.impute_values_[c] = X.select(nw.col(c).median()).item()

        return self


class MeanImputer(WeightColumnMixin, BaseImputer):
    """Transformer to impute missing values with the mean of the supplied columns.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weights_column : None or str, default = None
        Column containing weights.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mean) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @beartype
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> MeanImputer:
        """Calculate mean values to impute with from X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to "learn" the mean values from.

        y : None or pd.DataFrame or pd.Series, default = None
            Not required.

        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

            for c in self.columns:
                # filter out null rows so they don't count towards total weight
                filtered = X.filter(~nw.col(c).is_null())

                # calculate total weight and total of weighted col
                total_weight = filtered.select(nw.col(self.weights_column).sum()).item()
                total_weighted_col = filtered.select(
                    (nw.col(c) * nw.col(self.weights_column)).sum(),
                ).item()

                # find weighted mean and add to dict
                weighted_mean = total_weighted_col / total_weight

                self.impute_values_[c] = weighted_mean

        else:
            for c in self.columns:
                self.impute_values_[c] = X.select(nw.col(c).mean()).item()

        return self


class ModeImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the mode of the supplied columns.

    If mode is NaN, a warning will be raised.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weights_column : str
        Name of weights columns to use if mode should be in terms of sum of weights
        not count of rows.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mode) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @beartype
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> ModeImputer:
        """Calculate mode values to impute with from X - in the event of a tie,
        the highest modal value will be returned.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to "learn" the mode values from.

        y : None or pd/pl.DataFrame or pd/pl.Series, default = None
            Not required.

        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        if self.weights_column:
            # pull this out of loop to only check weights once
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)
            weights_column = self.weights_column

        else:
            weights_column = "dummy_unit_weights"
            native_backend = nw.get_native_namespace(X)
            X = X.with_columns(
                nw.new_series(
                    name=weights_column,
                    values=[1] * len(X),
                    backend=native_backend.__name__,
                ),
            )

        for c in self.columns:
            level_weights = (
                X.filter(~nw.col(c).is_null())
                .group_by(c)
                .agg(
                    nw.col(weights_column).sum(),
                )
            )

            if level_weights.is_empty():
                self.impute_values_[c] = None

                warnings.warn(
                    f"ModeImputer: The Mode of column {c} is None",
                    stacklevel=2,
                )

            else:
                max_weight = level_weights.select(nw.col(weights_column).max()).item()

                mode_values = level_weights.filter(
                    nw.col(weights_column) == max_weight,
                ).sort(by=c, descending=True)

                if len(mode_values) > 1:
                    warnings.warn(
                        f"ModeImputer: The Mode of column {c} is tied, will sort in descending order and return first candidate",
                        stacklevel=2,
                    )

                self.impute_values_[c] = mode_values.item(row=0, column=0)

        return self


class NearestMeanResponseImputer(BaseImputer):
    """Class to impute missing values with; the value for which the average response is closest
    to the average response for the unknown levels.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called. If the column does not contain nulls at fit,
        a warning will be issues and this transformer will have no effect on that column.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

    @beartype
    def fit(self, X: DataFrame, y: Series) -> NearestMeanResponseImputer:
        """Calculate mean values to impute with.

        Parameters
        ----------
        X : FrameT
            Data to fit the transformer on.

        y : nw.Series
            Response column used to determine the value to impute with. The average response for
            each level of every column is calculated. The level which has the closest average response
            to the average response of the unknown levels is selected as the imputation value.

        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        n_nulls = y.is_null().sum()

        if n_nulls > 0:
            msg = f"{self.classname()}: y has {n_nulls} null values"
            raise ValueError(msg)

        self.impute_values_ = {}

        X_y = nw.from_native(self._combine_X_y(X, y))
        response_column = "_temporary_response"

        for c in self.columns:
            c_nulls = X.select(nw.col(c).is_null())[c]

            if c_nulls.sum() == 0:
                msg = f"{self.classname()}: Column {c} has no missing values, this transformer will have no effect for this column."
                warnings.warn(msg, stacklevel=2)
                self.impute_values_[c] = None

            else:
                mean_response_by_levels = (
                    X_y.filter(~c_nulls).group_by(c).agg(nw.col(response_column).mean())
                )

                mean_response_nulls = X_y.filter(c_nulls)[response_column].mean()

                mean_response_by_levels = mean_response_by_levels.with_columns(
                    (nw.col(response_column) - mean_response_nulls)
                    .abs()
                    .alias("abs_diff_response"),
                )

                # take first value having the minimum difference in terms of average response
                self.impute_values_[c] = mean_response_by_levels.filter(
                    mean_response_by_levels["abs_diff_response"]
                    == mean_response_by_levels["abs_diff_response"].min(),
                )[c].item(index=0)

        return self


class NullIndicator(BaseTransformer):
    """Class to create a binary indicator column for null values.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to produce indicator columns for, if the default of None is supplied all columns in X are used
        when the transform method is called.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    """

    polars_compatible = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Create new columns indicating the position of null values for each variable in self.columns.

        Parameters
        ----------
        X : FrameT
            Data to add indicators to.

        """
        super().transform(X)

        X = _convert_dataframe_to_narwhals(X)

        for c in self.columns:
            X = X.with_columns(
                (nw.col(c).is_null()).alias(f"{c}_nulls"),
            )

        return X if not self.return_native else X.to_native()
