"""Contains transformers that apply string functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import narwhals as nw
import pandas as pd
from beartype import beartype
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.strings import (
    convert_string_columns_to_lowercase,
    extract_string_components,
    remove_characters_from_string_columns,
)
from tubular.types import DataFrame, GenericKwargs, ListOfOneStr, ListOfStrs


@register
class LowerCaseTransformer(BaseTransformer):
    """Transformer class to lower case of text columns.

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
    >>> from pprint import pprint
    >>> transformer = LowerCaseTransformer(
    ...     columns=["a"],
    ... )
    >>> transformer
    LowerCaseTransformer(columns=['a'])

    >>> json_dump = transformer.to_json()
    >>> pprint(json_dump)
    {'classname': 'LowerCaseTransformer',
     'fit': {},
     'init': {'columns': ['a'],
              'copy': False,
              'return_native': True,
              'verbose': False},
     'tubular_version': ...}

    >>> LowerCaseTransformer.from_json(json_dump)
    LowerCaseTransformer(columns=['a'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    def __init__(
        self,
        columns: str | ListOfStrs,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: Union[str, ListOfStrings]
            columns where values are to be lowercased.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return convert_string_columns_to_lowercase(columns=self.columns)

    def transform(self, X: DataFrame) -> DataFrame:
        """Lower case of text in given columns.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to lowercase.

        Returns
        -------
        X : DataFrame
            Transformed input X with text lowercased in given columns.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["HeLlO", None, "  HI"]})
        >>> transformer = LowerCaseTransformer(columns="a")
        >>> transformer.transform(test_df)
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ hello │
        │ null  │
        │   hi  │
        └───────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class ExtractStringComponentsTransformer(BaseTransformer):
    r"""Transformer class to extract components from string columns, split by given character.

    Attributes
    ----------
    by: str
        character to split on

    return_n_components: int
        number of components to return

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
    >>> from pprint import pprint
    >>> transformer = ExtractStringComponentsTransformer(
    ...     columns=["a"], by="@", return_n_components=2
    ... )
    >>> transformer
    ExtractStringComponentsTransformer(by='@', columns=['a'], return_n_components=2)

    >>> json_dump = transformer.to_json()
    >>> pprint(json_dump)
    {'classname': 'ExtractStringComponentsTransformer',
     'fit': {},
     'init': {'by': '@',
              'columns': ['a'],
              'copy': False,
              'return_n_components': 2,
              'return_native': True,
              'verbose': False},
     'tubular_version': ...}

    >>> ExtractStringComponentsTransformer.from_json(json_dump)
    ExtractStringComponentsTransformer(by='@', columns=['a'], return_n_components=2)

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        columns: str | ListOfStrs,
        by: str,
        return_n_components: int,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: Union[str, ListOfStrings]
            columns to remove characters from.

        by: str
            character to split strings by

        return_n_components:
            number of components to return

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.by = by
        self.return_n_components = return_n_components

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = ExtractStringComponentsTransformer(
        ...     columns=["a"], by="@", return_n_components=2
        ... )

        >>> transformer.get_feature_names_out()
        ['a_split_by_@_entry_0', 'a_split_by_@_entry_1']

        ```

        """
        return [
            f"{col}_split_by_{self.by}_entry_{i}"
            for col in self.columns
            for i in range(self.return_n_components)
        ]

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> from pprint import pprint
        >>> transformer = ExtractStringComponentsTransformer(
        ...     columns=["a"], by="@", return_n_components=2
        ... )

        >>> pprint(transformer.to_json())
        {'classname': 'ExtractStringComponentsTransformer',
         'fit': {},
         'init': {'by': '@',
                  'columns': ['a'],
                  'copy': False,
                  'return_n_components': 2,
                  'return_native': True,
                  'verbose': False},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["by"] = self.by
        json_dict["init"]["return_n_components"] = self.return_n_components

        return json_dict

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return extract_string_components(
            columns=self.columns,
            by=self.by,
            return_n_components=self.return_n_components,
        )

    def transform(self, X: DataFrame) -> DataFrame:
        r"""Extract components from string columns, split by given character.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to strip.

        Returns
        -------
        X : DataFrame
            Transformed input X with characters stripped from specified columns.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["greg@gmail.com", "bob@apple.net"]})
        >>> transformer = ExtractStringComponentsTransformer(
        ...     columns=["a"], by="@", return_n_components=2
        ... )
        >>> transformer.transform(test_df)
        shape: (2, 3)
        ┌────────────────┬──────────────────────┬──────────────────────┐
        │ a              ┆ a_split_by_@_entry_0 ┆ a_split_by_@_entry_1 │
        │ ---            ┆ ---                  ┆ ---                  │
        │ str            ┆ str                  ┆ str                  │
        ╞════════════════╪══════════════════════╪══════════════════════╡
        │ greg@gmail.com ┆ greg                 ┆ gmail.com            │
        │ bob@apple.net  ┆ bob                  ┆ apple.net            │
        └────────────────┴──────────────────────┴──────────────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class RemoveCharactersTransformer(BaseTransformer):
    r"""Transformer class to remove characters from text columns.

    Attributes
    ----------
    characters: list[str]
        list of characters to remove from text columns.

    characters_formatted: str
        characters attr formatted into regex string.

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
    >>> from pprint import pprint
    >>> transformer = RemoveCharactersTransformer(columns=["a"], characters=["\\d"])
    >>> transformer
    RemoveCharactersTransformer(characters=['\\d'], columns=['a'])

    >>> json_dump = transformer.to_json()
    >>> pprint(json_dump)
    {'classname': 'RemoveCharactersTransformer',
     'fit': {},
     'init': {'characters': ['\\d'],
              'columns': ['a'],
              'copy': False,
              'return_native': True,
              'verbose': False},
     'tubular_version': ...}

    >>> RemoveCharactersTransformer.from_json(json_dump)
    RemoveCharactersTransformer(characters=['\\d'], columns=['a'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        columns: str | ListOfStrs,
        characters: list[str],
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: Union[str, ListOfStrings]
            columns to remove characters from.

        characters: list[str]
            characters to remove from specified columns.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.characters = characters
        self.characters_formatted = r"[{}]".format("".join(self.characters))

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> from pprint import pprint
        >>> transformer = RemoveCharactersTransformer(columns=["a", "b"], characters=["a"])

        >>> pprint(transformer.to_json())
        {'classname': 'RemoveCharactersTransformer',
         'fit': {},
         'init': {'characters': ['a'],
                  'columns': ['a', 'b'],
                  'copy': False,
                  'return_native': True,
                  'verbose': False},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["characters"] = self.characters

        return json_dict

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return remove_characters_from_string_columns(
            columns=self.columns, characters_formatted=self.characters_formatted
        )

    def transform(self, X: DataFrame) -> DataFrame:
        r"""Strip unwanted characters from specified columns.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to strip.

        Returns
        -------
        X : DataFrame
            Transformed input X with characters stripped from specified columns.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["  8hi!", None, "9999hello  "]})
        >>> transformer = RemoveCharactersTransformer(columns=["a"], characters=["\W", "\s"])
        >>> transformer.transform(test_df)
        shape: (3, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ str       │
        ╞═══════════╡
        │ 8hi       │
        │ null      │
        │ 9999hello │
        └───────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If aspects of it have been useful to you, please raise an issue
    for it to be replaced with more specific transformers
    """,
)
class SeriesStrMethodTransformer(BaseTransformer):
    """Transformer that applies a pandas.Series.str method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.str method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Attributes
    ----------
    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pd.Series.str in transform.

    pd_method_name : str
        The name of the pd.Series.str method to call.

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

    deprecated: bool
        indicates if class has been deprecated

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    deprecated = True

    @beartype
    def __init__(
        self,
        new_column_name: str,
        pd_method_name: str,
        columns: ListOfOneStr,
        copy: bool = False,
        pd_method_kwargs: GenericKwargs | None = None,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class.

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

        copy: bool
            Perform transform on copy of df?

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.__init__().


        Raises
        ------
        AttributeError: if pd_method_name is not pd.Series method

        """
        super().__init__(columns=columns, copy=copy, **kwargs)

        if pd_method_kwargs is None:
            pd_method_kwargs = {}

        self.new_column_name = new_column_name
        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs

        try:
            ser = pd.Series(["a"])
            getattr(ser.str, pd_method_name)

        except Exception as err:
            msg = f'{self.classname()}: error accessing "str.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.str method'
            raise AttributeError(msg) from err

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply given pandas.Series.str method to given column.

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


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class StringConcatenator(BaseTransformer):
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

    deprecated: bool
        indicates if class has been deprecated

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    deprecated = True

    @beartype
    def __init__(
        self,
        columns: str | ListOfStrs,
        new_column_name: str = "new_column",
        separator: str = " ",
        **kwargs: bool,
    ) -> None:
        """Initialise class.

        Parameters
        ----------
        columns : str or list of str
            Columns to concatenate.
        new_column_name : str, default = "new_column"
            New column name
        separator : str, default = " "
            Separator for the new string value
        **kwargs:
            arguments for base class

        """
        super().__init__(columns=columns, **kwargs)

        self.new_column_name = new_column_name
        self.separator = separator

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

        # quick fix for empty frames, not spending much
        # time on this as transformer is deprecated
        if X.empty:
            X[self.new_column_name] = pd.Series(dtype=str)

        else:
            X[self.new_column_name] = (
                X[self.columns].astype(str).apply(self.separator.join, axis=1)
            )

        return X
