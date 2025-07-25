"""This module contains transformers that apply string functions."""

from __future__ import annotations

import pandas as pd

from tubular.base import BaseTransformer
from tubular.mixins import NewColumnNameMixin, SeparatorColumnMixin


class SeriesStrMethodTransformer(NewColumnNameMixin, BaseTransformer):
    """Tranformer that applies a pandas.Series.str method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.str method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Parameters
    ----------
    new_column_name : str
        The name of the column to be assigned to the output of running the pd.Series.str in transform.

    pd_method_name : str
        The name of the pandas.Series.str method to call e.g. 'split' or 'replace'

    columns : list
        Name of column to apply the transformer to. This needs to be passed as a list of length 1. Value passed
        in columns is saved in the columns attribute of the object. Note this has no default value so
        the user has to specify the column when initialising the transformer. This is to avoid all columns
        being picked up when super transform runs if the user forgets an input.

    pd_method_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.Series.str method when it is called.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pd.Series.str in transform.

    pd_method_name : str
        The name of the pd.Series.str method to call.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

    def __init__(
        self,
        new_column_name: str,
        pd_method_name: str,
        columns: list,
        copy: bool = False,
        pd_method_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, copy=copy, **kwargs)

        if len(columns) > 1:
            msg = f"{self.classname()}: columns arg should contain only 1 column name but got {len(columns)}"
            raise ValueError(msg)

        if type(pd_method_name) is not str:
            msg = f"{self.classname()}: unexpected type ({type(pd_method_name)}) for pd_method_name, expecting str"
            raise TypeError(msg)

        if pd_method_kwargs is None:
            pd_method_kwargs = {}
        else:
            if type(pd_method_kwargs) is not dict:
                msg = f"{self.classname()}: pd_method_kwargs should be provided as a dict or defaulted to None"
                raise TypeError(msg)

        for key in pd_method_kwargs:
            if type(key) is not str:
                msg = f"{self.classname()}: all keys in pd_method_kwargs must be a string value"
                raise TypeError(msg)

        self.check_and_set_new_column_name(new_column_name)

        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs

        try:
            ser = pd.Series(["a"])
            getattr(ser.str, pd_method_name)

        except Exception as err:
            msg = f'{self.classname()}: error accessing "str.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.str method'
            raise AttributeError(msg) from err

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specific column on input pandas.DataFrame (X) using the given pandas.Series.str method and
        assign the output back to column in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pd.Series.str method
        when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. These contain the output of
            running the pd.Series.str method.

        """
        X = super().transform(X)

        X[self.new_column_name] = getattr(X[self.columns[0]].str, self.pd_method_name)(
            **self.pd_method_kwargs,
        )

        return X


class StringConcatenator(NewColumnNameMixin, SeparatorColumnMixin, BaseTransformer):
    """Transformer to combine data from specified columns, of mixed datatypes, into a new column containing one string.

    Parameters
    ----------
    columns : str or list of str
        Columns to concatenate.
    new_column_name : str, default = "new_column"
        New column name
    separator : str, default = " "
        Separator for the new string value

    Attributes
    ----------
    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework
    """

    polars_compatible = False

    def __init__(
        self,
        columns: str | list[str],
        new_column_name: str = "new_column",
        separator: str = " ",
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        self.check_and_set_new_column_name(new_column_name)
        self.check_and_set_separator_column(separator)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Combine data from specified columns, of mixed datatypes, into a new column containing one string.

        Parameters
        ----------
        X : df
            Data to concatenate values on.

        Returns
        -------
        X : df
            Returns a dataframe with concatenated values.

        """
        X = super().transform(X)

        X[self.new_column_name] = (
            X[self.columns].astype(str).apply(self.separator.join, axis=1)
        )

        return X
