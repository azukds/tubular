"""Contains legacy transformers for introducing fixed columns and changing dtypes."""

from __future__ import annotations

from typing import Any, Optional, Union

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
from tubular.mixins import DropOriginalMixin
from tubular.types import (
    DataFrame,
    NonEmptyListOfStrs,
)


@register
class SetValueTransformer(BaseTransformer):
    """Transformer to set value of column(s) to a given value.

    This should be used if columns need to be set to a constant value.

    Attributes
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's
        supported functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to
        polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> SetValueTransformer(columns="a", value=1)
    SetValueTransformer(columns=['a'], value=1)

    ```

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
        value: Optional[Union[int, float, str, bool]],
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: list or str
            Columns to set values.

        value : various
            Value to set.

        **kwargs: bool
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        self.value = value

        super().__init__(columns=columns, **kwargs)

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
        >>> transformer = SetValueTransformer(columns="a", value=1)
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'SetValueTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'value': 1}, 'fit': {}}

        ```

        """  # noqa: E501
        json_dict = super().to_json()

        json_dict["init"]["value"] = self.value

        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Set columns to value.

        Parameters
        ----------
        X : DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : DataFrame
            Transformed input X with columns set to value.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = SetValueTransformer(columns="a", value=1)

        >>> test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        >>> transformer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i32 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 1   ┆ 5   │
        │ 1   ┆ 6   │
        └─────┴─────┘

        ```

        """
        X = super().transform(X, return_native_override=False)

        X = _convert_dataframe_to_narwhals(X)

        X = X.with_columns([nw.lit(self.value).alias(c) for c in self.columns])

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class RenameColumnsTransformer(BaseTransformer, DropOriginalMixin):
    """Transformer to rename a given set of columns.

    This can be useful for personalising the auto-output names from
    other transformers, or for creating a few different versions
    of a given column to undergo separate paths of logic in a pipeline
    (as the expression logic effectively creates duplicates of the column).

    Attributes
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's
        supported functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to
        polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> transformer = RenameColumnsTransformer(
    ...     columns="a", new_column_names={"a": "new_a"}
    ... )  # noqa: E501
    >>> transformer
    RenameColumnsTransformer(columns=['a'], new_column_names={'a': 'new_a'})

    >>> # transformer can also be dumped to json and reinitialised

    >>> json_dump = transformer.to_json()
    >>> json_dump  # doctest: +NORMALIZE_WHITESPACE
    {'tubular_version': ...,
    'classname': 'RenameColumnsTransformer',
    'init': {'columns': ['a'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'new_column_names': {'a': 'new_a'},
    'drop_original': True},
    'fit': {}}

    >>> RenameColumnsTransformer.from_json(json_dump)
    RenameColumnsTransformer(columns=['a'], new_column_names={'a': 'new_a'})

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[
            NonEmptyListOfStrs,
            str,
        ],
        new_column_names: dict[str, str],
        drop_original: bool = True,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns: list or str
            Columns to set values.

        new_column_names: dict[str, str]
            dictionary mapping provided columns to updated names

        drop_original: bool
            indicates whether to drop original columns.

        **kwargs: bool
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        Raises
        ------
            ValueError: if provided columns are not keys of
                provided new_column_names

        """
        super().__init__(columns=columns, **kwargs)

        msg = f"{self.classname()}: all provided columns must appear as keys in new_column_names"  # noqa: E501
        for column in self.columns:
            if column not in new_column_names:
                raise ValueError(msg)

        self.new_column_names = new_column_names
        self.drop_original = drop_original

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = RenameColumnsTransformer(
        ...     columns=["a", "b"],
        ...     new_column_names={"a": "new_a", "b": "new_b"},
        ... )

        >>> transformer.get_feature_names_out()
        ['new_a', 'new_b']

        ```

        """
        return list(self.new_column_names.values())

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
        >>> transformer = RenameColumnsTransformer(
        ...     columns="a", new_column_names={"a": "new_a"}
        ... )  # noqa: E501
        >>> transformer.to_json()  # doctest: +NORMALIZE_WHITESPACE
        {'tubular_version': ...,
        'classname': 'RenameColumnsTransformer',
        'init': {'columns': ['a'],
        'copy': False,
        'verbose': False,
        'return_native': True,
        'new_column_names': {'a': 'new_a'},
        'drop_original': True},
        'fit': {}}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "new_column_names": self.new_column_names,
                "drop_original": self.drop_original,
            }
        )

        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Create column copies.

        Parameters
        ----------
        X : DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : DataFrame
            Transformed input X with columns set to value.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = RenameColumnsTransformer(
        ...     columns="a", new_column_names={"a": "new_a"}
        ... )  # noqa: E501

        >>> test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        >>> transformer.transform(test_df)
        shape: (3, 2)
        ┌─────┬───────┐
        │ b   ┆ new_a │
        │ --- ┆ ---   │
        │ i64 ┆ i64   │
        ╞═════╪═══════╡
        │ 4   ┆ 1     │
        │ 5   ┆ 2     │
        │ 6   ┆ 3     │
        └─────┴───────┘

        ```

        """
        X = super().transform(X, return_native_override=False)

        X = _convert_dataframe_to_narwhals(X)

        X = X.with_columns(
            [nw.col(c).alias(self.new_column_names[c]) for c in self.columns]
        )

        X = DropOriginalMixin.drop_original_column(X, self.drop_original, self.columns)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class ColumnDtypeSetter(BaseTransformer):
    """Transformer to set transform columns in a dataframe to a dtype.

    Attributes
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json,
        which limits it's supported functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to
        polars/pandas agnostic narwhals framework

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

    FITS = False

    jsonable = False

    deprecated = True

    def __init__(
        self,
        columns: str | list[str],
        dtype: type | str,
        **kwargs: dict[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list
            Columns to set dtype. Must be set or transform will not run.

        dtype : type or string
            dtype object to set columns to or a string interpretable as one
            by pd.api.types.pandas_dtype e.g. float or 'float'

        **kwargs: dict[str, Any]
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns, **kwargs)

        self.__validate_dtype(dtype)

        self.dtype = dtype

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data.

        Parameters
        ----------
        X: pd.DataFrame
            data to transform.

        Returns
        -------
            pd.DataFrame: transformed data

        """
        X = super().transform(X)

        X[self.columns] = X[self.columns].astype(self.dtype)

        return X

    def __validate_dtype(self, dtype: str) -> None:
        """Check string is a valid dtype.

        Raises
        ------
            TypeError: for invalid pandas dtype

        """
        try:
            pd.api.types.pandas_dtype(dtype)
        except TypeError:
            msg = f"{self.classname()}: data type '{dtype}' not understood as a valid dtype"  # noqa: E501
            raise TypeError(msg) from None
