"""Contains transformers that other transformers in the package inherit from.

These transformers contain key checks to be applied in all cases.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import narwhals as nw
import pandas as pd
from beartype import beartype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _get_version,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.mixins import DropOriginalMixin
from tubular.types import (
    DataFrame,
    GenericKwargs,
    NonEmptyListOfStrs,
    Series,
)

pd.options.mode.copy_on_write = True


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base tranformer class which all other transformers in the package inherit from.

    Provides fit and transform methods (required by sklearn transformers), simple input checking
    and functionality to copy X prior to transform.

    Attributes:
    ----------
    columns : list
        Either a list of str values giving which columns in a input pandas.DataFrame the transformer
        will be applied to.

    copy : bool
        Should X be copied before tansforms are applied? Copy argument no longer used and will be deprecated in a future release

    verbose : bool
        Print statements to show which methods are being run or not.

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
    -------
    >>> BaseTransformer(
    ... columns='a',
    ...    )
    BaseTransformer(columns=['a'])

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = True

    _version = _get_version()

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
        columns: Union[
            NonEmptyListOfStrs,
            str,
        ],
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ) -> None:
        """Init method for class.

        Parameters
        ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object.

        copy : bool, default = False
            Should X be copied before tansforms are applied? Copy argument no longer used and will be deprecated in a future release

        verbose : bool, default = False
            Should statements be printed when methods are run?

        return_native: bool, default = True
            Controls whether transformer returns narwhals or native pandas/polars type

        """
        self.verbose = verbose

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
        >>> transformer  = BaseTransformer(
        ... columns='a',
        ...    )

        >>> transformer.get_feature_names_out()
        ['a']

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
            RuntimeError: if transformer does not have to/from json functionality enabled

        Examples
        --------
            >>> transformer=BaseTransformer(columns=['a', 'b'])

            >>> # version will vary for local vs CI, so use ... as generic match
            >>> transformer.to_json()
            {'tubular_version': ..., 'classname': 'BaseTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True}, 'fit': {}}

        """
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
            "fit": {},
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
            RuntimeError: if transformer does not have to/from json functionality enabled

        Examples
        --------
            >>> json_dict={
            ... 'init': {
            ...         'columns' :['a','b']
            ...         },
            ... 'fit': {}
            ... }

            >>> BaseTransformer.from_json(json=json_dict)
            BaseTransformer(columns=['a', 'b'])

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
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> BaseTransformer:
        """Check data before fit.

        Fit calls the columns_check method which will check that the columns attribute is set and all values are present in X

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        Returns
        -------
            BaseTransformer: returns self

        Examples
        --------
            >>> import polars as pl
            >>> transformer=BaseTransformer(
            ... columns='a',
            ...    )
            >>> df=pl.DataFrame({'a': [1,2], 'b': [3,4]})
            >>> transformer.fit(df)
            BaseTransformer(columns=['a'])

        """
        if self.verbose:
            print("BaseTransformer.fit() called")

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        self.columns_check(X)

        return self

    @block_from_json
    @nw.narwhalify
    def _combine_X_y(self, X: DataFrame, y: nw.Series) -> DataFrame:
        """Combine X and y by adding a new column with the values of y to a copy of X.

        The new column response column will be called `_temporary_response`.

        This method can be used by transformers that need to use the response, y, together
        with the explanatory variables, X, in their `fit` methods.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data containing explanatory variables.

        y : pd/pl.Series
            Response variable.

        Returns
        -------
            pd/pl/nw.DataFrame: DataFrame with added column containing y

        Examples
        --------
            # correct usage
            >>> import polars as pl
            >>> transformer=BaseTransformer(
            ... columns='a',
            ...    )
            >>> X=pl.DataFrame({'a': [1,2], 'b': [3,4]})
            >>> y=pl.Series(name='a', values=[1,2])
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
            >>> X=pl.DataFrame({'a': [1,2], 'b': [3,4]})
            >>> y=pl.Series(name='a', values=[1])
            >>> transformer._combine_X_y(X, y)
            Traceback (most recent call last):
            ...
            narwhals.exceptions.InvalidOperationError: Series _temporary_response, length 1 doesn't match the DataFrame height of 2
            ...

        """
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        return X.with_columns(_temporary_response=y)

    @beartype
    def _process_return_native(self, return_native_override: Optional[bool]) -> bool:
        """Determine whether to override return_native attr.

        Parameters
        ----------
        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        bool: whether or not to return native type

        Example:
        --------
        >>> transformer=BaseTransformer(
        ... columns='a',
        ... return_native=True
        ... )

        >>> transformer._process_return_native(return_native_override=False)
        False

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
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Check data before child transform.

        Transform calls the columns_check method which will check columns in columns attribute are in X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to transform with the transformer.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : pd/pl.DataFrame
            Input X, copied if specified by user.

        Examples
        --------
            >>> import polars as pl
            >>> transformer=BaseTransformer(
            ... columns='a',
            ...    )

            >>> df=pl.DataFrame({'a': [1,2], 'b': [3,4]})

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

        """
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
        >>> transformer=BaseTransformer(
        ... columns='a',
        ...    )

        >>> transformer.check_is_fitted('columns')

        """
        check_is_fitted(self, attribute)

    @beartype
    def columns_check(self, X: DataFrame) -> None:
        """Check that the columns attribute is set and all values are present in X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to check columns are in.

        Raises
        ------
        ValueError: if columns missing from dataframe

        Examples
        --------
        >>> import polars as pl
        >>> transformer=BaseTransformer(
        ... columns='a',
        ...    )

        >>> df=pl.DataFrame({'a': [1,2], 'b': [3,4]})

        >>> transformer.columns_check(df)

        """
        X = _convert_dataframe_to_narwhals(X)

        missing_columns = set(self.columns).difference(X.columns)
        if len(missing_columns) != 0:
            msg = f"{self.classname()}: variables {missing_columns} not in X"
            raise ValueError(
                msg,
            )


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has been deprecated in favour of more specialised transformers.
    See the aggregations module for aggregation type functionality formerly covered by
    this transformer.
    If other functionality was being used from this transformer, then please submit an
    issue for it to be redeveloped!
    """,
)
class DataFrameMethodTransformer(DropOriginalMixin, BaseTransformer):
    """Tranformer that applies a pandas.DataFrame method.

    Transformer assigns the output of the method to a new column or columns. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.DataFrame method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Attributes
    ----------
    new_column_names : str or list of str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        new_column_names: Union[list[str], str],
        pd_method_name: str,
        columns: Optional[Union[list[str], str]],
        pd_method_kwargs: Optional[GenericKwargs] = None,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        """Init method for class.

        Parameters
        ----------
        new_column_names : str or list of str
            The name of the column or columns to be assigned to the output of running the
            pandas method in transform.

        pd_method_name : str
            The name of the pandas.DataFrame method to call.

        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. This is avoid likely
            when the user forget to set columns, in this case all columns would be picked up when super
            transform runs.

        pd_method_kwargs : dict, default = {}
            A dictionary of keyword arguments to be passed to the pd.DataFrame method when it is called.

        drop_original : bool, default = False
            Should original columns be dropped?

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.__init__().

        Raises
        ------
        AttributeError: if pd_method_name is not valid pd.DataFrame method

        """
        super().__init__(columns=columns, **kwargs)

        if pd_method_kwargs is None:
            pd_method_kwargs = {}

        self.new_column_names = new_column_names
        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs
        self.drop_original = drop_original

        try:
            df = pd.DataFrame()
            getattr(df, pd_method_name)

        except Exception as err:
            msg = f'{self.classname()}: error accessing "{pd_method_name}" method on pd.DataFrame object - pd_method_name should be a pd.DataFrame method'
            raise AttributeError(msg) from err

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data.

        Uses the given pandas.DataFrame method and assign the output
        back to column or columns in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pandas DataFrame method when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.new_column_names) added. These contain the output of
            running the pandas DataFrame method.

        """
        X = super().transform(X)

        # quick fix for empty frames, not spending much
        # time on this as transformer is deprecated.
        # the new_column_names attr is a bit messy,
        # sometimes str and sometimes list
        # editing init to make it always a list
        # broke other tests which didn't seem worth fixing
        # so have included handling for both cases here..
        if X.empty:
            # hard to know the best dtype to use here given the
            # flexibility of this transformer, which is
            # partially why it was deprecated
            if isinstance(self.new_column_names, list):
                for col in X[self.new_column_names]:
                    X[col] = pd.Series(dtype=float)
            else:
                X[self.new_column_names] = pd.Series(dtype=float)

        else:
            X[self.new_column_names] = getattr(X[self.columns], self.pd_method_name)(
                **self.pd_method_kwargs,
            )

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
        )
