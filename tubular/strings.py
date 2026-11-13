"""Contains transformers that apply string functions."""

from __future__ import annotations

from typing import Any

import narwhals as nw
from beartype import beartype

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.strings import (
    convert_string_columns_to_lowercase,
    extract_string_components,
    indicate_if_string_columns_contain_reference,
    remove_characters_from_string_columns,
)
from tubular.types import (
    DataFrame,
    ListOfStrs,
    StrictlyPositiveInt,
)


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
     'fit': {'is_fitted_': False},
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

    @beartype
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

        X = super().transform(X, return_native_override=False)

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
     'fit': {'is_fitted_': False},
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
        return_n_components: StrictlyPositiveInt,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: Union[str, ListOfStrings]
            columns containing string values to split into components.

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
         'fit': {'is_fitted_': False},
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
            Data containing columns to extract components from.

        Returns
        -------
        X : DataFrame
            Transformed input X with string components extracted from columns.

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

        X = super().transform(X, return_native_override=False)

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
     'fit': {'is_fitted_': False},
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
         'fit': {'is_fitted_': False},
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

        X = super().transform(X, return_native_override=False)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class StringContainsTransformer(BaseTransformer):
    r"""Transformer class to indicate if given columns contain reference values.

    Attributes
    ----------
    reference: str
        column or value to compare against, e.g.
        look for values of reference='a' in columns ['b', 'c'].

    reference_as_column: bool
        indicates whether reference represents a column (or value).
        Note, reference_as_column=True is not supported for pandas backend.

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
    >>> transformer = StringContainsTransformer(
    ...     columns=["a"], reference="b", reference_as_column=True
    ... )
    >>> transformer
    StringContainsTransformer(columns=['a'], reference='b',
                              reference_as_column=True)

    >>> json_dump = transformer.to_json()
    >>> pprint(json_dump)
    {'classname': 'StringContainsTransformer',
     'fit': {'is_fitted_': False},
     'init': {'columns': ['a'],
              'copy': False,
              'reference': 'b',
              'reference_as_column': True,
              'return_native': True,
              'verbose': False},
     'tubular_version': ...}

    >>> StringContainsTransformer.from_json(json_dump)
    StringContainsTransformer(columns=['a'], reference='b',
                              reference_as_column=True)

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
        reference: str,
        reference_as_column: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: Union[str, ListOfStrings]
            columns to remove characters from.

        reference: str
            reference value to search for

        reference_as_column: bool
            whether to treat reference as a column or a literal

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.reference = reference
        self.reference_as_column = reference_as_column

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = StringContainsTransformer(columns=["a", "b"], reference="c")

        >>> transformer.get_feature_names_out()
        ['a_contains_c', 'b_contains_c']

        ```

        """
        return [f"{col}_contains_{self.reference}" for col in self.columns]

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
        >>> transformer = StringContainsTransformer(
        ...     columns=["a"], reference="b", reference_as_column=True
        ... )

        >>> pprint(transformer.to_json())
        {'classname': 'StringContainsTransformer',
         'fit': {'is_fitted_': False},
         'init': {'columns': ['a'],
                  'copy': False,
                  'reference': 'b',
                  'reference_as_column': True,
                  'return_native': True,
                  'verbose': False},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["reference"] = self.reference
        json_dict["init"]["reference_as_column"] = self.reference_as_column

        return json_dict

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return indicate_if_string_columns_contain_reference(
            columns=self.columns,
            reference=self.reference,
            reference_as_column=self.reference_as_column,
        )

    def transform(self, X: DataFrame) -> DataFrame:
        r"""Indicate if provided columns contain reference values.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to strip.

        Returns
        -------
        X : DataFrame
            Transformed input X with characters stripped from specified columns.

        Raises
        ------
        TypeError: if called on pandas df when reference_as_column=True

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame(
        ...     {"a": ["cat", "dog", None, "mouse"], "b": ["cat", "rat", None, "mouse"]}
        ... )
        >>> transformer = StringContainsTransformer(
        ...     columns=["a"], reference="b", reference_as_column=True
        ... )
        >>> transformer.transform(test_df)
        shape: (4, 3)
        ┌───────┬───────┬──────────────┐
        │ a     ┆ b     ┆ a_contains_b │
        │ ---   ┆ ---   ┆ ---          │
        │ str   ┆ str   ┆ bool         │
        ╞═══════╪═══════╪══════════════╡
        │ cat   ┆ cat   ┆ true         │
        │ dog   ┆ rat   ┆ false        │
        │ null  ┆ null  ┆ null         │
        │ mouse ┆ mouse ┆ true         │
        └───────┴───────┴──────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        backend = nw.get_native_namespace(X).__name__

        if backend == "pandas" and self.reference_as_column:
            msg = f"{self.classname()}: reference_as_column=True is only supported for polars backend"
            raise TypeError(msg)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, self.return_native)
