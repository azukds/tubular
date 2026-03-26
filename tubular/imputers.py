"""Contains transformers that deal with imputation of missing values."""

from __future__ import annotations

import warnings
from typing import Any

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
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _is_null,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.mixins import WeightColumnMixin
from tubular.types import DataFrame, ListOfStrs, NumericTypes, Series

pl.enable_string_cache()


@register
class BaseImputer(BaseTransformer):
    """Contains transform method that will use fill nulls with values from self.impute_values_.

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

    Examples
    --------
    ```pycon
    >>> BaseImputer(columns=["a", "b"])
    BaseImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    # this class is not by itself jsonable, as needs attrs
    # which are set in the child classes
    jsonable = False

    FITS = False

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Raises
        ------
        RuntimeError:
            if class is not jsonable

        Examples
        --------
        ```pycon
        >>> arbitrary_imputer = ArbitraryImputer(columns=["a", "b"], impute_value=1)

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> arbitrary_imputer.to_json()
        {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 1}, 'fit': {'impute_values_': {'a': 1, 'b': 1}}}

        >>> mean_imputer = MeanImputer(columns=["a", "b"])

        >>> test_df = pl.DataFrame({"a": [1, None], "b": [None, 2]})

        >>> _ = mean_imputer.fit(test_df)

        >>> mean_imputer.to_json()
        {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 1.0, 'b': 2.0}}}

        ```

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
        """Update input expressions to include imputation.

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

    def _check_for_failed_fit(self) -> None:
        """Check if fit failed to find needed attrs (if impute_values_ are None).

        Raises
        ------
        ValueError: if impute_values_ have come out as None

        """
        failed_columns = [
            col for col in self.columns if _is_null(self.impute_values_[col])
        ]

        if failed_columns:
            msg = f"fit has failed for columns {failed_columns}, it is possible that all rows are invalid - check for null/negative weights, all null columns, or other invalid conditions listed in the docstring"
            raise ValueError(msg)

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Impute missing values with values calculated from fit method.

        Parameters
        ----------
        X : DataFrame
            Data to impute.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the median value for the specified columns.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> imputer = BaseImputer(columns=["a", "b"])

        >>> imputer.impute_values_ = {"a": 2, "b": 3.5}

        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})

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

        ```

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


class _NumberImputer(BaseImputer):
    """Private subclass to handle arbitrary number imputation.

    Attributes
    ----------
    impute_value : bool
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

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: int | float,
        columns: str | list[str],
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : int or float
            Value to impute nulls with.
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = _NumberImputer(columns=["a", "b"], impute_value=5)
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

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        schema = X.schema

        bad_types = [
            schema[col]
            for col in self.columns
            if schema[col] not in {*NumericTypes, nw.Unknown}
        ]

        if bad_types:
            msg = f"""
                ArbitraryImputer: transformer can only handle Float/Int/UInt/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        X = BaseTransformer.transform(self, X, return_native_override=False)

        # next handle imputing
        transform_expressions = {
            col: self._generate_imputation_expressions(
                nw.col(col),
                col,
            )
            for col in self.columns
        }

        X = X.with_columns(**transform_expressions) if transform_expressions else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class _StringImputer(BaseImputer):
    """Private subclass to handle arbitrary string imputation.

    Attributes
    ----------
    impute_value : bool
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

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: str,
        columns: str | list[str],
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : str
            Value to impute nulls with.
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

    def cat_to_enum_expr(self, expr: nw.Expr, categories: list[str]) -> nw.Expr:
        """Update expression to include handling of category types.

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

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["cat", None, "dog"]})
        >>> imputer = _StringImputer(columns=["a"], impute_value="missing")
        >>> imputer = imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ cat     │
        │ missing │
        │ dog     │
        └─────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        schema = X.schema
        native_namespace = nw.get_native_namespace(X).__name__

        X = BaseTransformer.transform(self, X, return_native_override=False)

        schema = X.schema

        bad_types = [
            schema[col]
            for col in self.columns
            if schema[col] not in {nw.String, nw.Categorical, nw.Enum, nw.Unknown}
        ]

        if bad_types:
            msg = f"""
                ArbitraryImputer: transformer can only handle String/Categorical/Enum/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        transform_expressions = {}
        for col in self.columns:
            # have to handle categorical vars for pandas upfront
            if native_namespace == "pandas":
                transform_expressions[col] = (
                    self.cat_to_enum_expr(
                        nw.col(col),
                        categories=X.get_column(col).cat.get_categories().to_list(),
                    )
                    if ((schema[col] == nw.Categorical) or (schema[col] == nw.Enum))
                    else nw.col(col)
                )
            else:
                transform_expressions[col] = nw.col(col)

            # next handle imputing
            transform_expressions[col] = self._generate_imputation_expressions(
                transform_expressions[col],
                col,
            )

        X = X.with_columns(**transform_expressions) if transform_expressions else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class _BooleanImputer(BaseImputer):
    """Private subclass to handle arbitrary boolean imputation.

    Attributes
    ----------
    impute_value : bool
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

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: bool,
        columns: str | list[str],
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : bool
            Value to impute nulls with.
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [True, None, False]})
        >>> imputer = _BooleanImputer(columns=["a"], impute_value=True)
        >>> imputer = imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ false │
        └───────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        schema = X.schema

        allowed_types = [nw.Boolean, nw.Unknown]
        allowed_types_str = "Boolean/Unknown"

        if nw.get_native_namespace(X).__name__ == "pandas":
            allowed_types += [nw.Object]
            allowed_types_str += "/Object"

        bad_types = [
            schema[col] for col in self.columns if schema[col] not in allowed_types
        ]

        if bad_types:
            msg = f"""
                ArbitraryImputer: transformer can only handle {allowed_types_str} type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        X = BaseTransformer.transform(self, X, return_native_override=False)

        # next handle imputing
        transform_expressions = {
            col: self._generate_imputation_expressions(
                nw.col(col),
                col,
            ).cast(nw.Boolean)
            for col in self.columns
        }

        X = X.with_columns(**transform_expressions) if transform_expressions else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class ArbitraryImputer(BaseImputer):
    """Transformer to impute null values with an arbitrary pre-defined value.

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

    Examples
    --------
    ```pycon
    >>> arbitrary_imputer = ArbitraryImputer(columns=["a", "b"], impute_value=5)
    >>> arbitrary_imputer
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = arbitrary_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 5}, 'fit': {'impute_values_': {'a': 5, 'b': 5}}}

    >>> ArbitraryImputer.from_json(json_dump)
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: int | float | str | bool,
        columns: str | list[str],
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : int or float or str or bool
            Value to impute nulls with.
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Example:
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = ArbitraryImputer(columns=["a", "b"], impute_value=5)
        >>> imputer = imputer.fit(test_df)
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

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        if isinstance(self.impute_value, (int, float)) and not isinstance(
            self.impute_value, bool
        ):
            imp = _NumberImputer(
                columns=self.columns,
                impute_value=self.impute_value,
                return_native=self.return_native,
            )

        elif isinstance(self.impute_value, str):
            imp = _StringImputer(
                columns=self.columns,
                impute_value=self.impute_value,
                return_native=self.return_native,
            )

        else:
            imp = _BooleanImputer(
                columns=self.columns,
                impute_value=self.impute_value,
                return_native=self.return_native,
            )

        return imp.transform(X)


@register
class MedianImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the median of the supplied columns.

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

    Examples
    --------
    ```pycon
    >>> median_imputer = MedianImputer(
    ...     columns=["a", "b"],
    ... )
    >>> median_imputer
    MedianImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = median_imputer.fit(test_df)

    >>> json_dump = median_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MedianImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MedianImputer.from_json(json_dump)
    MedianImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.

        weights_column: None or str, default=None
            Column containing weights

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | None = None) -> MedianImputer:
        """Calculate median values to impute with from X.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the median values from.

        y : Series or None, default = None
            Not required.

        Returns
        -------
            MedianImputer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = MedianImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
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

        ```

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
            valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
                self.weights_column, self.verbose
            )
            X = X.filter(valid_weights_filter_expr)

            for c in not_all_null_columns:
                col_not_null_expr = ~nw.col(c).is_null()

                X = X.sort(c)

                col_expr = nw.col(c).filter(col_not_null_expr)
                weight_expr = nw.col(self.weights_column).filter(col_not_null_expr)

                median_expr = _get_median_calculation_expression(
                    initial_column_expr=col_expr,
                    initial_weights_expr=weight_expr,
                )

                # impute value is weighted median
                self.impute_values_[c] = X.select(median_expr).item(0, 0)

        else:
            median_exprs = {
                c: _get_median_calculation_expression(nw.col(c), None)
                for c in not_all_null_columns
            }
            results_dict = X.select(
                **median_exprs,
            ).to_dict(as_series=False)

            self.impute_values_.update(
                {col: value[0] for col, value in results_dict.items()},
            )

        self._check_for_failed_fit()

        return self


@register
class MeanImputer(WeightColumnMixin, BaseImputer):
    """Transformer to impute missing values with the mean of the supplied columns.

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

    Examples
    --------
    ```pycon
    >>> mean_imputer = MeanImputer(
    ...     columns=["a", "b"],
    ... )
    >>> mean_imputer
    MeanImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = mean_imputer.fit(test_df)

    >>> json_dump = mean_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MeanImputer.from_json(json_dump)
    MeanImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.

        weights_column : None or str, default = None
            Column containing weights.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | None = None) -> MeanImputer:
        """Calculate mean values to impute with from X.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the mean values from.

        y : Series or None, default = None
            Not required.

        Returns
        -------
            MeanImputer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = MeanImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
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

        ```

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
        valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
            weights_column, self.verbose
        )
        X = X.filter(valid_weights_filter_expr)

        weighted_mean_exprs = _get_mean_calculation_expressions(
            self.columns,
            weights_column,
        )

        results_dict = X.select(**weighted_mean_exprs).to_dict(as_series=False)

        # results looks like {key: [value]} so extract value from list
        self.impute_values_.update(
            {col: value[0] for col, value in results_dict.items()},
        )

        self._check_for_failed_fit()

        return self


@register
class ModeImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the mode of the supplied columns.

    If mode is NaN, a warning will be raised.

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

    Examples
    --------
    ```pycon
    >>> mode_imputer = ModeImputer(
    ...     columns=["a", "b"],
    ... )
    >>> mode_imputer
    ModeImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = mode_imputer.fit(test_df)

    >>> json_dump = mode_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ModeImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None}, 'fit': {'impute_values_': {'a': 0, 'b': 1}}}

    >>> ModeImputer.from_json(json_dump)
    ModeImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

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

        """
        super().__init__(columns=columns, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | None = None) -> ModeImputer:
        """Calculate mode values to impute with from X.

        In the event of a tie, the highest modal value will be returned.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the mode values from.

        y : Series or None, default = None
            Not required.

        Returns
        -------
        ModeImputer:
            fitted class instance

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = ModeImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
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

        ```

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
        valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
            weights_column, self.verbose
        )
        X = X.filter(valid_weights_filter_expr)

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

        self._check_for_failed_fit()

        return self


@register
class NullIndicator(BaseTransformer):
    """Class to create a binary indicator column for null values.

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

    Examples
    --------
    ```pycon
    >>> null_indicator = NullIndicator(
    ...     columns=["a", "b"],
    ... )
    >>> null_indicator
    NullIndicator(columns=['a', 'b'])

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = null_indicator.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'NullIndicator', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True}, 'fit': {}}

    >>> NullIndicator.from_json(json_dump)
    NullIndicator(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: ListOfStrs | str,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to produce indicator columns for, if the default of None is supplied all columns in X are used
            when the transform method is called.

        kwargs: bool
            arguments for base class, e.g. verbose.

        """
        super().__init__(columns=columns, **kwargs)

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Create new columns indicating the position of null values for each variable in self.columns.

        Parameters
        ----------
        X : DataFrame
            Data to add indicators to.

        Returns
        -------
        DataFrame:
            dataframe with null indicator columns added

        Examples
        --------
        --------,
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
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

        ```

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
    """Impute nulls with the value where the average target is most similar to that for the nulls.

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

    deprecated: bool
        indicates if class has been deprecated

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = False

    FITS = True

    deprecated = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called. If the column does not contain nulls at fit,
            a warning will be issues and this transformer will have no effect on that column.

        kwargs: bool
            arguments for base class, e.g. verbose

        """
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

        Returns
        -------
        NearestMeanResponseImputer: fitted class instance

        Raises
        ------
        ValueError: provided y contains nulls

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
