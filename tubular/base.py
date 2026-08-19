"""Contains transformers that other transformers in the package inherit from.

These transformers contain key checks to be applied in all cases.
"""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import pandas as pd
from beartype import beartype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tubular._utils import (
    _collect_series,
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _get_version,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.types import (
    DataFrame,
    LazyFrame,
    ListOfStrs,
    Series,
)

pd.options.mode.copy_on_write = True

FEATURE_REGISTRY = {}

CLASS_REGISTRY = {}


def register(cls: BaseTransformer) -> BaseTransformer:
    """Add transformer to registry dict.

    Returns:
    -------
    cls - transformer

    Example:
    -------
    ```pycon
    >>> @register
    ... class MyTransformer(BaseTransformer):
    ...     pass
    ...
    >>> CLASS_REGISTRY["MyTransformer"]
    <class 'tubular.base.MyTransformer'>

    ```

    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


@register
class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base transformer class which all other transformers in the package inherit from.

    Provides fit and transform methods (required by sklearn transformers), simple input
    checking and functionality to copy X prior to transform.

    Attributes:
    ----------
    columns : list
        Either a list of str values giving which columns in a input pandas.DataFrame the
        transformer will be applied to.

    copy : bool
        Should X be copied before transforms are applied?
        Copy argument no longer used and will be deprecated in a future release

    verbose : bool
        Print statements to show which methods are being run or not.

    built_from_json: bool
        indicates if transformer was reconstructed from json,
        which limits it's supported functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to
        polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> BaseTransformer(
    ...     columns="a",
    ... )
    BaseTransformer(columns=['a'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = True

    _version = _get_version()

    def __init_subclass__(cls: BaseTransformer) -> None:
        """Logic to be run when a new child class is defined.

        This populates a dictionary, which will help us track which
        transformers in the repo support which functionality.
        """
        deprecated = getattr(cls, "deprecated", False)

        # ignore deprecated transformers and base classes
        if (
            deprecated
            or cls.__name__.startswith("Base")
            or cls.__name__.startswith("_")
        ):
            return

        FEATURE_REGISTRY[cls.__name__] = {
            "polars_compatible": cls.polars_compatible,
            # repo was originally written in pandas, so the is a given
            "pandas_compatible": True,
            "jsonable": cls.jsonable,
            "lazyframe_compatible": cls.lazyframe_compatible,
            "extractable_exprs": cls.extractable_exprs,
        }

    def classname(self) -> str:
        """Return the name of the current class when called.

        Returns
        -------
            str: name of class

        """
        return type(self).__name__

    @beartype
    def __init__(
        self,
        columns: ListOfStrs | str,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
        drop_original: bool | None = None,
    ) -> None:
        """Init method for class.

        Parameters
        ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into
            a list.
            Value passed in columns is saved in the columns attribute on the object.

        copy : bool, default = False
            Should X be copied before transforms are applied?
            Copy argument no longer used and will be deprecated in a future release

        verbose : bool, default = False
            Should statements be printed when methods are run?

        return_native: bool, default = True
            Controls whether transformer returns narwhals or native pandas/polars type

        drop_original: bool or None, default=None
            deprecated argument, has no effect.

        """
        self.verbose = verbose
        self.drop_original = drop_original

        if self.drop_original:
            warnings.warn(
                f"{self.classname()}: drop_original argument has "
                "been deprecated and will have no effect",
                stacklevel=2,
            )

        if self.verbose:
            print("BaseTransformer.__init__() called")

        # make sure columns is a single str or list of strs
        if isinstance(columns, str):
            self.columns = [columns]

        elif isinstance(columns, list):
            self.columns = columns

        self.copy = copy
        self.return_native = return_native

        self.built_from_json = False
        self.is_fitted_ = False

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Child classes will need to overload this method if their behaviour is
        more complex than just returning the input columns.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )

        >>> transformer.get_feature_names_out()
        ['a']

        ```

        """
        return self.columns

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
        RuntimeError: if transformer does not have to/from json functionality
            enabled

        Examples
        --------
        ```pycon
        >>> transformer = BaseTransformer(columns=["a", "b"])

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True}, 'fit': {'is_fitted_': False}}

        ```

        """  # noqa: E501
        if not self.jsonable:
            msg = (
                "This transformer has not yet had to/from json functionality developed"
            )
            raise RuntimeError(
                msg,
            )

        return {
            "tubular_version": self._version,
            "classname": self.classname(),
            "init": {
                "columns": self.columns,
                "copy": self.copy,
                "verbose": self.verbose,
                "return_native": self.return_native,
            },
            "fit": {"is_fitted_": self.is_fitted_},
        }

    @classmethod
    def from_json(cls, json: dict[str, Any]) -> BaseTransformer:
        """Rebuild transformer from json dict, readyfor transform.

        Parameters
        ----------
        json: dict[str, dict[str, Any]]
            json-ified transformer

        Returns
        -------
        BaseTransformer:
            reconstructed transformer class, ready for transform

        Raises
        ------
        RuntimeError: if transformer does not have to/from json
            functionality enabled

        Examples
        --------
        ```pycon
        >>> json_dict = {"init": {"columns": ["a", "b"]}, "fit": {}}

        >>> BaseTransformer.from_json(json=json_dict)
        BaseTransformer(columns=['a', 'b'])

        ```

        """
        if not cls.jsonable:
            msg = (
                "This transformer has not yet had to/from json functionality developed"
            )
            raise RuntimeError(
                msg,
            )
        instance = cls(**json["init"])

        for attr in json["fit"]:
            setattr(instance, attr, json["fit"][attr])

        instance.built_from_json = True

        return instance

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | LazyFrame | None = None) -> BaseTransformer:
        """Check data before fit.

        Fit calls the columns_check method which will check that the columns
        attribute is set and all values are present in X

        Parameters
        ----------
        X : DataFrame
            Data to fit the transformer on.

        y : None or Series or LazyFrame, default = None
            Optional argument only required for the transformer to work with sklearn
            pipelines.

        Returns
        -------
            BaseTransformer: returns self

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )
        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> transformer.fit(df)
        BaseTransformer(columns=['a'])

        ```

        """
        if self.verbose:
            print("BaseTransformer.fit() called")

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        self.columns_check(X)
        return self

    @block_from_json
    @beartype
    def _combine_X_y(
        self, X: DataFrame, y: Series | LazyFrame, return_native_override: bool = True
    ) -> DataFrame:
        """Combine X and y by adding a new column with the values of y to a copy of X.

        The new column response column will be called `_temporary_response`.

        This method can be used by transformers that need to use the response, y,
        together with the explanatory variables, X, in their `fit` methods.

        Parameters
        ----------
        X : DataFrame
            Data containing explanatory variables.

        y : Series or LazyFrame
            Response variable.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
            DataFrame: DataFrame with added column containing y

        Examples
        --------
        ```pycon
        # correct usage
        >>> import polars as pl
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )
        >>> X = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> y = pl.Series(name="a", values=[1, 2])
        >>> transformer._combine_X_y(X, y)
        shape: (2, 3)
        ┌─────┬─────┬─────────────────────┐
        │ a   ┆ b   ┆ _temporary_response │
        │ --- ┆ --- ┆ ---                 │
        │ i64 ┆ i64 ┆ i64                 │
        ╞═════╪═════╪═════════════════════╡
        │ 1   ┆ 3   ┆ 1                   │
        │ 2   ┆ 4   ┆ 2                   │
        └─────┴─────┴─────────────────────┘

        # example error from mismatched X/y
        >>> X = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> y = pl.Series(name="a", values=[1])
        >>> transformer._combine_X_y(X, y)
        Traceback (most recent call last):
        ...
        narwhals.exceptions.InvalidOperationError: Series _temporary_response, length 1 doesn't match the DataFrame height of 2
        ...

        ```

        """  # noqa: E501
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        return_native = self._process_return_native(return_native_override)

        # If both X and y are LazyFrames, use join to maintain lazy evaluation
        if isinstance(X, nw.LazyFrame) and isinstance(y, nw.LazyFrame):
            # Convert LazyFrame y to LazyFrame with row index for joining
            y_named = y.with_row_index("__row_idx__")
            X_indexed = X.with_row_index("__row_idx__")
            y_col = y.columns[0]
            X = (
                X_indexed.join(
                    y_named.select("__row_idx__", y_col), on="__row_idx__", how="inner"
                )
                .select("*")
                .exclude("__row_idx__")
                .rename({y_col: "_temporary_response"})
            )
        elif isinstance(y, nw.LazyFrame):
            # If y is LazyFrame but X is not, collect y first
            y = _collect_series(y)
            X = X.with_columns(_temporary_response=y)
        else:
            # For eager frames or Series, use with_columns
            X = X.with_columns(_temporary_response=y)

        return _return_narwhals_or_native_dataframe(X, return_native)

    @beartype
    def _process_return_native(self, return_native_override: bool | None) -> bool:
        """Determine whether to override return_native attr.

        Parameters
        ----------
        return_native_override: Optional[bool]
            option to override return_native attr in transformer,
            useful when calling parent methods

        Returns
        -------
        bool: whether or not to return native type

        Example:
        --------
        ```pycon
        >>> transformer = BaseTransformer(columns="a", return_native=True)

        >>> transformer._process_return_native(return_native_override=False)
        False

        ```

        """
        return (
            return_native_override
            if return_native_override is not None
            else self.return_native
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Check data before child transform.

        Transform calls the columns_check method which will check columns in columns
        attribute are in X.

        Parameters
        ----------
        X : DataFrame
            Data to transform with the transformer.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer,
            useful when calling parent methods

        Returns
        -------
        X : DataFrame
            Input X, copied if specified by user.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )

        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        >>> transformer.transform(df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        ```

        """
        self.check_is_fitted("is_fitted_")
        return_native = self._process_return_native(return_native_override)

        X = _convert_dataframe_to_narwhals(X)

        if self.copy and not isinstance(X, nw.LazyFrame):
            # to prevent overwriting original dataframe
            X = X.clone()

        self.columns_check(X)

        if self.verbose:
            print("BaseTransformer.transform() called")

        return _return_narwhals_or_native_dataframe(X, return_native)

    def check_is_fitted(self, attribute: str) -> None:
        """Check if particular attributes are on the object.

        This is useful to do before running transform to avoid
        trying to transform data without first running the fit method.

        Wrapper for utils.validation.check_is_fitted function.

        Parameters
        ----------
        attribute : List
            List of str values giving names of attribute to check exist on self.

        Example
        -------
        ```pycon
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )

        >>> transformer.check_is_fitted("columns")

        ```

        """
        check_is_fitted(self, attribute)

    @beartype
    def columns_check(self, X: DataFrame) -> None:
        """Check that the columns attribute is set and all values are present in X.

        Parameters
        ----------
        X : DataFrame
            Data to check columns are in.

        Raises
        ------
        ValueError: if columns missing from dataframe

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = BaseTransformer(
        ...     columns="a",
        ... )

        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        >>> transformer.columns_check(df)

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        missing_columns = set(self.columns).difference(X.collect_schema().names())
        if len(missing_columns) != 0:
            msg = f"{self.classname()}: variables {missing_columns} not in X"
            raise ValueError(
                msg,
            )
