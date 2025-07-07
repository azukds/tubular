"""This module contains transformers that other transformers in the package inherit
from. These transformers contain key checks to be applied in all cases.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import narwhals as nw
import pandas as pd
from beartype import beartype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tubular._types import GenericKwargs, NonEmptyListOfStrs  # noqa: TCH001
from tubular.mixins import DropOriginalMixin

if TYPE_CHECKING:
    from narwhals.typing import FrameT

pd.options.mode.copy_on_write = True


class BaseTransformer(TransformerMixin, BaseEstimator):
    """Base tranformer class which all other transformers in the package inherit from.

    Provides fit and transform methods (required by sklearn transformers), simple input checking
    and functionality to copy X prior to transform.

    Parameters
    ----------
    columns : None or list or str
        Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object.

    copy : bool, default = True
        Should X be copied before tansforms are applied? Copy argument no longer used and will be deprecated in a future release

    verbose : bool, default = False
        Should statements be printed when methods are run?

    Attributes
    ----------
    columns : list
        Either a list of str values giving which columns in a input pandas.DataFrame the transformer
        will be applied to.

    copy : bool
        Should X be copied before tansforms are applied? Copy argument no longer used and will be deprecated in a future release

    verbose : bool
        Print statements to show which methods are being run or not.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    def classname(self) -> str:
        """Method that returns the name of the current class when called."""
        return type(self).__name__

    @beartype
    def __init__(
        self,
        columns: Union[
            NonEmptyListOfStrs,
            str,
        ],
        copy: Union[bool, None] = None,
        verbose: bool = False,
    ) -> None:
        if copy is not None:
            warnings.warn(
                "copy argument no longer used and will be deprecated in a future release",
                DeprecationWarning,
                stacklevel=2,
            )

        self.verbose = verbose

        if self.verbose:
            print("BaseTransformer.__init__() called")

        # make sure columns is a single str or list of strs
        if isinstance(columns, str):
            self.columns = [columns]

        elif isinstance(columns, list):
            self.columns = columns

        self.copy = copy

    @nw.narwhalify
    def fit(self, X: FrameT, y: nw.Series | None = None) -> BaseTransformer:
        """Base transformer fit method, checks X and y types. Currently only pandas DataFrames are allowed for X
        and DataFrames or Series for y.

        Fit calls the columns_check method which will check that the columns attribute is set and all values are present in X

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """
        if self.verbose:
            print("BaseTransformer.fit() called")

        self.columns_check(X)

        if not X.shape[0] > 0:
            msg = f"{self.classname()}: X has no rows; {X.shape}"
            raise ValueError(msg)

        if y is not None:
            if not isinstance(y, nw.Series):
                msg = f"{self.classname()}: unexpected type for y, should be a polars or pandas Series"
                raise TypeError(msg)

            if not y.shape[0] > 0:
                msg = f"{self.classname()}: y is empty; {y.shape}"
                raise ValueError(msg)

        return self

    @nw.narwhalify
    def _combine_X_y(self, X: FrameT, y: nw.Series) -> FrameT:
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

        """
        if not isinstance(X, (nw.DataFrame, nw.LazyFrame)):
            msg = f"{self.classname()}: X should be a polars or pandas DataFrame/LazyFrame"
            raise TypeError(msg)

        if not isinstance(y, nw.Series):
            msg = f"{self.classname()}: y should be a polars or pandas Series"
            raise TypeError(msg)

        if X.shape[0] != y.shape[0]:
            msg = f"{self.classname()}: X and y have different numbers of rows ({X.shape[0]} vs {y.shape[0]})"
            raise ValueError(msg)

        return X.with_columns(_temporary_response=y)

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Base transformer transform method; checks X type (pandas/polars DataFrame only) and copies data if requested.

        Transform calls the columns_check method which will check columns in columns attribute are in X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to transform with the transformer.

        Returns
        -------
        X : pd/pl.DataFrame
            Input X, copied if specified by user.

        """
        self.columns_check(X)

        if self.verbose:
            print("BaseTransformer.transform() called")

        # to prevent overwriting original dataframe
        X_view = X.clone()

        if not X.shape[0] > 0:
            msg = f"{self.classname()}: X has no rows; {X.shape}"
            raise ValueError(msg)

        return X_view

    def check_is_fitted(self, attribute: str) -> None:
        """Check if particular attributes are on the object. This is useful to do before running transform to avoid
        trying to transform data without first running the fit method.

        Wrapper for utils.validation.check_is_fitted function.

        Parameters
        ----------
        attributes : List
            List of str values giving names of attribute to check exist on self.

        """
        check_is_fitted(self, attribute)

    @nw.narwhalify
    def columns_check(self, X: FrameT) -> None:
        """Method to check that the columns attribute is set and all values are present in X.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to check columns are in.

        """
        if not isinstance(X, (nw.DataFrame, nw.LazyFrame)):
            msg = f"{self.classname()}: X should be a polars or pandas DataFrame/LazyFrame"
            raise TypeError(msg)

        if not isinstance(self.columns, list):
            msg = f"{self.classname()}: self.columns should be a list"
            raise TypeError(msg)

        for c in self.columns:
            if c not in X.columns:
                raise ValueError(f"{self.classname()}: variable " + c + " is not in X")


class DataFrameMethodTransformer(DropOriginalMixin, BaseTransformer):

    """Tranformer that applies a pandas.DataFrame method.

    Transformer assigns the output of the method to a new column or columns. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.DataFrame method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

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

    Attributes
    ----------
    new_column_names : str or list of str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

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
        """Transform input pandas DataFrame (X) using the given pandas.DataFrame method and assign the output
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
