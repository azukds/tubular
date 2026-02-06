"""Contains legacy transformers for introducing fixed columns and changing dtypes."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional, Union

import narwhals as nw
from beartype import beartype
from beartype.vale import Is

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
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
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        X = X.with_columns([nw.lit(self.value).alias(c) for c in self.columns])

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class SimpleCastDtypes(str, Enum):
    """Allowed dtypes for ColumnDtypeSetter."""

    FLOAT64 = "Float64"
    FLOAT32 = "Float32"
    INT64 = "Int64"
    INT32 = "Int32"
    INT16 = "Int16"
    INT8 = "Int8"
    UINT64 = "UInt64"
    UINT32 = "UInt32"
    UINT16 = "UInt16"
    UINT8 = "UInt8"
    BOOLEAN = "Boolean"
    STRING = "String"
    CATEGORICAL = "Categorical"


SimpleCastDtypesStr = Annotated[
    str,
    Is[lambda s: s in SimpleCastDtypes._value2member_map_],
]


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

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    deprecated = False

    @beartype
    def __init__(
        self,
        columns: Union[str, NonEmptyListOfStrs],
        dtype: SimpleCastDtypesStr,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : Union[str, NonEmptyListOfStrs]
            Columns to set dtype. Must be set or transform will not run.

        dtype : SimpleCastDtypesStr
            dtype to set column to

        **kwargs: dict[str, Any]
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns, **kwargs)

        self.dtype = dtype

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
        >>> transformer = ColumnDtypeSetter(columns="a", dtype="Float32")
        >>> pprint(transformer.to_json(), sort_dicts=True)
        {'classname': 'ColumnDtypeSetter',
         'fit': {},
         'init': {'columns': ['a'],
                  'copy': False,
                  'dtype': 'Float32',
                  'return_native': True,
                  'verbose': False},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["dtype"] = self.dtype

        return json_dict

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform data.

        Parameters
        ----------
        X: DataFrame
            data to transform.

        Returns
        -------
            DataFrame: transformed data

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2]})
        >>> transformer = ColumnDtypeSetter(columns="a", dtype="Float32")
        >>> transformer.transform(df)
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f32 │
        ╞═════╡
        │ 1.0 │
        │ 2.0 │
        └─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        backend = nw.get_native_namespace(X).__name__

        X = super().transform(X, return_native_override=False)

        if backend == "pandas" and self.dtype == "Boolean":
            X = X.with_columns(
                nw.maybe_convert_dtypes(X[col]).cast(nw.Boolean) for col in self.columns
            )

        else:
            X = X.with_columns(
                [nw.col(col).cast(getattr(nw, self.dtype)) for col in self.columns]
            )

        return _return_narwhals_or_native_dataframe(X, self.return_native)
