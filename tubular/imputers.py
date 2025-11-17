"""This module contains transformers that deal with imputation of missing values."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Union

import narwhals as nw
import polars as pl
from beartype import beartype
from typing_extensions import deprecated

from tubular._checks import _get_all_null_columns
from tubular._stats import (
    _get_mean_calculation_expressions,
    _get_median_calculation_expression,
    _get_mode_calculation_expressions,
)
from tubular._utils import (
    _assess_pandas_object_column,
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer
from tubular.mixins import WeightColumnMixin
from tubular.types import DataFrame, NonEmptyListOfStrs, Series

pl.enable_string_cache()


class BaseImputer(BaseTransformer):
    """Base imputer class containing standard transform method that will use pd.Series.fillna with the
    values in the impute_values_ attribute.

    Other imputers in this module should inherit from this class.

    Attributes
    ----------

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> BaseImputer(columns=["a", "b"])
    BaseImputer(columns=['a', 'b'])

    """

    polars_compatible = True

    lazyframe_compatible = False

    # this class is not by itself jsonable, as needs attrs
    # which are set in the child classes
    jsonable = False

    FITS = False

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """dump transformer to json dict

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------

        >>> arbitrary_imputer=ArbitraryImputer(columns=['a', 'b'], impute_value=1)

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> arbitrary_imputer.to_json()
        {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 1}, 'fit': {'impute_values_': {'a': 1, 'b': 1}}}

        >>> mean_imputer=MeanImputer(columns=['a', 'b'])

        >>> test_df=pl.DataFrame({'a': [1, None],  'b': [None, 2]})

        >>> _ = mean_imputer.fit(test_df)

        >>> mean_imputer.to_json()
        {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 1.0, 'b': 2.0}}}

        """
        if not self.jsonable:
            msg = (
                "This transformer has not yet had to/from json functionality developed"
            )
            raise RuntimeError(
                msg,
            )

        self.check_is_fitted("impute_values_")

        json_dict = super().to_json()

        # slightly awkward here as API not fully shared
        # across classes
        if isinstance(
            self,
            (
                MeanImputer,
                MedianImputer,
                ModeImputer,
            ),
        ):
            json_dict["init"]["weights_column"] = self.weights_column
        elif isinstance(self, ArbitraryImputer):
            json_dict["init"]["impute_value"] = self.impute_value

        json_dict["fit"]["impute_values_"] = self.impute_values_

        return json_dict

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

        Example:
        --------
        >>> import polars as pl

        >>> imputer = BaseImputer(columns=["a", "b"])

        >>> imputer.impute_values_= {"a":2, "b":3.5}

        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})

        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 3.0 │
        │ 2   ┆ 3.5 │
        │ 2   ┆ 4.0 │
        └─────┴─────┘
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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples:
    --------
    >>> arbitrary_imputer = ArbitraryImputer(
    ... columns=["a", "b"], impute_value= 5
    ... )
    >>> arbitrary_imputer
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump=arbitrary_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 5}, 'fit': {'impute_values_': {'a': 5, 'b': 5}}}

    >>> ArbitraryImputer.from_json(json_dump)
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)
    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

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

        return expr.cast(nw.Enum({*categories, self.impute_value}))

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

        Example:
        --------
        >>> import polars as pl
        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})
        >>> imputer = ArbitraryImputer(columns=["a", "b"], impute_value= 5)
        >>> imputer= imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 5   ┆ 5   │
        │ 2   ┆ 4   │
        └─────┴─────┘

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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> median_imputer = MedianImputer(
    ... columns=["a", "b"],
    ... )
    >>> median_imputer
    MedianImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df=pl.DataFrame({'a': [0, None], 'b': [None, 1]})

    >>> _ = median_imputer.fit(test_df)

    >>> json_dump=median_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MedianImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MedianImputer.from_json(json_dump)
    MedianImputer(columns=['a', 'b'])
    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> MedianImputer:
        """Calculate median values to impute with from X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to "learn" the median values from.

        y : None or pd/pl.Series, default = None
            Not required.

        Example:
        --------
        >>> import polars as pl
        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})
        >>> imputer = MedianImputer(columns=["a", "b"])
        >>> imputer= imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 3.0 │
        │ 1.5 ┆ 3.5 │
        │ 2.0 ┆ 4.0 │
        └─────┴─────┘
        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        all_null_cols = _get_all_null_columns(X, self.columns)

        if all_null_cols:
            # touch the dict entry for each all null col so that they are recorded
            self.impute_values_.update(
                dict.fromkeys(all_null_cols),
            )

            warnings.warn(
                f"{self.classname()}: The Median of columns {all_null_cols} will be None",
                stacklevel=2,
            )

        not_all_null_columns = sorted(set(self.columns).difference(set(all_null_cols)))

        # as median depends on data ordering, it is less amenable to writing in
        # pure expression form, so implementation here is still
        # slightly pandas-like
        # also, the weighted median approach is genuinely different to the unweighted
        # approach, so have left as two separate logic flows
        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)
            for c in not_all_null_columns:
                X = X.sort(c).filter(~nw.col(c).is_null())

                median_expr = _get_median_calculation_expression(
                    c,
                    self.weights_column,
                )

                # impute value is weighted median
                self.impute_values_[c] = X.select(median_expr).item(0, 0)

        else:
            median_exprs = {
                c: _get_median_calculation_expression(c, None)
                for c in not_all_null_columns
            }
            results_dict = X.select(
                **median_exprs,
            ).to_dict(as_series=False)

            self.impute_values_.update(
                {col: value[0] for col, value in results_dict.items()},
            )

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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> mean_imputer = MeanImputer(
    ... columns=["a", "b"],
    ... )
    >>> mean_imputer
    MeanImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df=pl.DataFrame({'a': [0, None], 'b': [None, 1]})

    >>> _ = mean_imputer.fit(test_df)

    >>> json_dump=mean_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MeanImputer.from_json(json_dump)
    MeanImputer(columns=['a', 'b'])
    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> MeanImputer:
        """Calculate mean values to impute with from X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to "learn" the mean values from.

        y : None or pd.DataFrame or pd.Series, default = None
            Not required.

        Example:
        --------
        >>> import polars as pl
        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})
        >>> imputer = MeanImputer(columns=["a", "b"])
        >>> imputer= imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 3.0 │
        │ 1.5 ┆ 3.5 │
        │ 2.0 ┆ 4.0 │
        └─────┴─────┘
        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        native_backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=native_backend.__name__,
                return_native=False,
            )

        WeightColumnMixin.check_weights_column(self, X, weights_column)

        weighted_mean_exprs = _get_mean_calculation_expressions(
            self.columns,
            weights_column,
        )

        results_dict = X.select(**weighted_mean_exprs).to_dict(as_series=False)

        # results looks like {key: [value]} so extract value from list
        self.impute_values_.update(
            {col: value[0] for col, value in results_dict.items()},
        )

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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> mode_imputer = ModeImputer(
    ... columns=["a", "b"],
    ... )
    >>> mode_imputer
    ModeImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df=pl.DataFrame({'a': [0, None], 'b': [None, 1]})

    >>> _ = mode_imputer.fit(test_df)

    >>> json_dump=mode_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ModeImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0, 'b': 1}}}

    >>> ModeImputer.from_json(json_dump)
    ModeImputer(columns=['a', 'b'])
    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

    @block_from_json
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

        Example:
        --------
        >>> import polars as pl
        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})
        >>> imputer = ModeImputer(columns=["a", "b"])
        >>> imputer= imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=backend.__name__,
                return_native=False,
            )

        WeightColumnMixin.check_weights_column(self, X, weights_column)

        self.impute_values_ = {}

        all_null_cols = _get_all_null_columns(X, self.columns)

        if all_null_cols:
            # touch the dict entry for each all null col so that they are recorded
            self.impute_values_.update(
                dict.fromkeys(all_null_cols),
            )

            warnings.warn(
                f"{self.classname()}: The Mode of columns {all_null_cols} will be None",
                stacklevel=2,
            )

        not_all_null_columns = sorted(set(self.columns).difference(set(all_null_cols)))

        mode_value_exprs = _get_mode_calculation_expressions(
            not_all_null_columns,
            weights_column,
        )

        results_dict = X.select(**mode_value_exprs).to_dict(as_series=True)

        for c in results_dict:
            mode_values = results_dict[c]

            mode_values = mode_values.drop_nulls().sort(
                descending=True,
            )

            n_mode_vals = len(mode_values)

            if n_mode_vals > 1:
                warnings.warn(
                    f"ModeImputer: The Mode of column {c} is tied, will sort in descending order and return first candidate",
                    stacklevel=2,
                )

            self.impute_values_[c] = mode_values.item(0)

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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> null_indicator = NullIndicator(
    ... columns=["a", "b"],
    ... )
    >>> null_indicator
    NullIndicator(columns=['a', 'b'])

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump=null_indicator.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'NullIndicator', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True}, 'fit': {}}

    >>> NullIndicator.from_json(json_dump)
    NullIndicator(columns=['a', 'b'])
    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[
            NonEmptyListOfStrs,
            str,
        ],
        **kwargs: Optional[bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Create new columns indicating the position of null values for each variable in self.columns.

        Parameters
        ----------
        X : FrameT
            Data to add indicators to.

        Example:
        --------
        >>> import polars as pl
        >>> test_df = pl.DataFrame({'a': [1, None, 2], 'b': [3, None, 4]})
        >>> imputer = NullIndicator(columns=["a", "b"])
        >>> imputer.transform(test_df)
        shape: (3, 4)
        ┌──────┬──────┬─────────┬─────────┐
        │ a    ┆ b    ┆ a_nulls ┆ b_nulls │
        │ ---  ┆ ---  ┆ ---     ┆ ---     │
        │ i64  ┆ i64  ┆ bool    ┆ bool    │
        ╞══════╪══════╪═════════╪═════════╡
        │ 1    ┆ 3    ┆ false   ┆ false   │
        │ null ┆ null ┆ true    ┆ true    │
        │ 2    ┆ 4    ┆ false   ┆ false   │
        └──────┴──────┴─────────┴─────────┘
        """
        X = super().transform(X, return_native_override=False)

        X = _convert_dataframe_to_narwhals(X)

        X = X.with_columns(
            (nw.col(c).is_null()).alias(f"{c}_nulls") for c in self.columns
        )

        return X if not self.return_native else X.to_native()


# DEPRECATED TRANSFORMERS


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
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

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = False

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

        if (n_nulls := y.is_null().sum()) > 0:
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
