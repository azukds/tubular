"""This module contains transformers that other transformers in the package inherit
from. These transformers contain key checks to be applied in all cases.
"""

from __future__ import annotations

import warnings

import pandas as pd
import narwhals as nw
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tubular._version import __version__

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

    version_ : str
        Version number (__version__ attribute from _version.py).

    """

    def classname(self) -> str:
        """Method that returns the name of the current class when called."""
        return type(self).__name__

    def __init__(
        self,
        columns: list[str] | str,
        copy: bool | None = None,
        verbose: bool = False,
    ) -> None:
        self.version_ = __version__

        if not isinstance(verbose, bool):
            msg = f"{self.classname()}: verbose must be a bool"
            raise TypeError(msg)

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
            if not len(columns) > 0:
                msg = f"{self.classname()}: columns has no values"
                raise ValueError(msg)

            for c in columns:
                if not isinstance(c, str):
                    msg = f"{self.classname()}: each element of columns should be a single (string) column name"
                    raise TypeError(msg)

            self.columns = columns

        else:
            msg = f"{self.classname()}: columns must be a string or list with the columns to be pre-processed (if specified)"
            raise TypeError(msg)

        self.copy = copy

    def fit(self,
            X: pd.DataFrame | pl.DataFrame | pl.LazyFrame | nw.DataFrame | nw.LazyFrame,
            y: pd.Series | pl.Series | None = None) -> BaseTransformer:
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

        if not (isinstance(X, nw.DataFrame) | isinstance(X, nw.LazyFrame)):
            X = nw.from_native(X[:])  # [:] creates a view for pandas to prevent updates to original frame

        if not set(self.columns).issubset(X.columns):
            disjoint_columns = list(set(self.columns) - set(X.columns))
            raise ValueError(f"{self.classname()}: variable " + ', '.join(disjoint_columns) + " is not in X")

        if not X.shape[0] > 0:
            msg = f"{self.classname()}: X has no rows; {X.shape}"
            raise ValueError(msg)

        if y is not None:
            if not (isinstance(y, pd.Series) | isinstance(y, pl.Series)):
                msg = (
                    f"{self.classname()}: unexpected type for y, should be a pd.Series or pl.Series"
                )
                raise TypeError(msg)

            # TODO - are we free of the copy problem now?

            if not y.shape[0] > 0:
                msg = f"{self.classname()}: y is empty; {y.shape}"
                raise ValueError(msg)

        return self

    def _combine_X_y(self,
                     X: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                     y: pd.Series | pl.Series) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        """Combine X and y by adding a new column with the values of y to a copy of X.

        The new column response column will be called `_temporary_response`.

        This method can be used by transformers that need to use the response, y, together
        with the explanatory variables, X, in their `fit` methods.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing explanatory variables.

        y : pd.Series
            Response variable.

        """

        if not (isinstance(X, nw.DataFrame) | isinstance(X, nw.LazyFrame)):
            X = nw.from_native(X[:])  # [:] creates a view for pandas frames to prevent updates to original frame

        y_assigned_to_X = X.with_columns(pl.Series(name="_temporary_response", values=y))

        return nw.to_native(y_assigned_to_X)

    def transform(self,
                  X: pd.DataFrame | pl.DataFrame | pl.LazyFrame | nw.DataFrame | nw.LazyFrame) ->\
                  pd.DataFrame | pl.DataFrame | pl.LazyFrame | nw.DataFrame | nw.LazyFrame:
        """Base transformer transform method; checks X type (pandas DataFrame only) and copies data if requested.

        Transform will check that pre-specified columns for transform are present in X.

        Parameters
        ----------
        X : Dataframe to transform with the transformer.

        Returns
        -------
        X : Input X, validated

        """

        if not set(self.columns).issubset(X.columns):
            disjoint_columns = list(set(self.columns) - set(X.columns))
            raise ValueError(f"{self.classname()}: variable " + ', '.join(disjoint_columns) + " is not in X")

        if self.verbose:
            print("BaseTransformer.transform() called")

        if not X.shape[0] > 0:
            msg = f"{self.classname()}: X has no rows; {X.shape}"
            raise ValueError(msg)

        return X

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

    @staticmethod
    def check_weights_column(X: pd.DataFrame | pl.DataFrame | pl.LazyFrame | nw.DataFrame | nw.LazyFrame,
                             weights_column: str) -> None:
        """Helper method for validating weights column in dataframe.

        Args:
        ----
            X (pd.DataFrame): df containing weight column
            weights_column (str): name of weight column

        """
        # Cannot stay lazy
        nw_frame = nw.DataFrame(X)

        if weights_column not in nw_frame.columns:
          msg = f"weight col ({weights_column}) is not present in columns of data"
          raise ValueError(msg)

        nw_numeric_types = [nw.Int64, nw.Int32, nw.Int16, nw.Int8,
                            nw.UInt64, nw.UInt32, nw.UInt16, nw.UInt8,
                            nw.Float64, nw.Float32]

        if nw_frame[weights_column].dtype not in nw_numeric_types:
            msg = "weight column must be numeric."
            raise ValueError(msg)


        if (nw_frame[weights_column] < 0).sum() != 0:
            msg = "weight column must be positive"
            raise ValueError(msg)

        # check weight non-null
        if nw_frame[weights_column].is_null().sum() != 0:
            msg = "weight column must be non-null"
            raise ValueError(msg)


class DataFrameMethodTransformer(BaseTransformer):
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

    """

    def __init__(
        self,
        new_column_names: list[str] | str,
        pd_method_name: str,
        columns: list[str] | str | None,
        pd_method_kwargs: dict[str, object] | None = None,
        drop_original: bool = False,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if type(new_column_names) is list:
            for i, item in enumerate(new_column_names):
                if type(item) is not str:
                    msg = f"{self.classname()}: if new_column_names is a list, all elements must be strings but got {type(item)} in position {i}"
                    raise TypeError(msg)

        elif type(new_column_names) is not str:
            msg = f"{self.classname()}: unexpected type ({type(new_column_names)}) for new_column_names, must be str or list of strings"
            raise TypeError(msg)

        if type(pd_method_name) is not str:
            msg = f"{self.classname()}: unexpected type ({type(pd_method_name)}) for pd_method_name, expecting str"
            raise TypeError(msg)

        if pd_method_kwargs is None:
            pd_method_kwargs = {}
        else:
            if type(pd_method_kwargs) is not dict:
                msg = f"{self.classname()}: pd_method_kwargs should be a dict but got type {type(pd_method_kwargs)}"
                raise TypeError(msg)

            for i, k in enumerate(pd_method_kwargs.keys()):
                if type(k) is not str:
                    msg = f"{self.classname()}: unexpected type ({type(k)}) for pd_method_kwargs key in position {i}, must be str"
                    raise TypeError(msg)

        if type(drop_original) is not bool:
            msg = f"{self.classname()}: unexpected type ({type(drop_original)}) for drop_original, expecting bool"
            raise TypeError(msg)

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

        if self.drop_original:
            X = X.drop(self.columns, axis=1)

        return X
