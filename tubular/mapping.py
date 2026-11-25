"""Contains transformers that apply different types of mappings to columns."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl
from beartype import beartype

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _filter_column_dict,
    _return_narwhals_or_native_dataframe,
    _sort_dict,
    _sort_nested_dict,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.mapping import (
    RETURN_DTYPES,
    map_boolean_columns,
    map_categorical_columns,
    map_float_columns,
    map_integer_columns,
    map_string_columns,
)
from tubular.types import PANDAS_NULLABLE_TYPES, DataFrame


@register
class BaseMappingTransformer(BaseTransformer):
    """Base Transformer Extension for mapping transformers.

    Attributes
    ----------
    mappings : dict
        Dictionary of mappings for each column individually. The dict passed to mappings in
        init is set to the mappings attribute.

    mappings_from_null: dict[str, Any]
        dict storing what null values will be mapped to. Generally best to use an imputer,
        but this functionality is useful for inverting pipelines.

    return_dtypes: dict[str, RETURN_DTYPES]
        Dictionary of col:dtype for returned columns

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

    Examples
    --------
    ```pycon
    >>> BaseMappingTransformer(
    ...     mappings={"a": {"Y": 1, "N": 0}},
    ...     return_dtypes={"a": "Int8"},
    ... )
    BaseMappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                           return_dtypes={'a': 'Int8'})

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        mappings: dict[str, dict[Any, Any]],
        return_dtypes: dict[str, RETURN_DTYPES] | None = None,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        mappings : dict
            Dictionary containing column mappings. Each value in mappings should be a dictionary
            of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
            example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
            a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

        return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
            Dictionary of col:dtype for returned columns

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        Raises
        ------
        ValueError:
            if multiple mappings for null values are provided

        """
        mappings_from_null = dict.fromkeys(mappings)
        for col, col_mappings in mappings.items():
            null_keys = [key for key in col_mappings if pd.isna(key)]

            if len(null_keys) > 1:
                multi_null_map_msg = f"Multiple mappings have been provided for null values in column {col}, transformer is set up to handle nan/None/NA as one"
                raise ValueError(
                    multi_null_map_msg,
                )

            # Assign the mapping to the single null key if it exists
            if len(null_keys) != 0:
                mappings_from_null[col] = col_mappings[null_keys[0]]

        self.mappings = mappings

        self.mappings_from_null = mappings_from_null

        columns = list(mappings.keys())

        # if return_dtypes is not provided, then infer from mappings
        if return_dtypes is not None:
            provided_return_dtype_keys = set(return_dtypes.keys())
        else:
            return_dtypes = {}
            provided_return_dtype_keys = set()

        for col in set(mappings.keys()).difference(provided_return_dtype_keys):
            return_dtypes[col] = self._infer_return_type(col)

        self.return_dtypes = return_dtypes

        super().__init__(columns=columns, **kwargs)
        self.is_fitted_ = True  # Does not fit

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
        >>> mapping_transformer = BaseMappingTransformer(mappings={"a": {"x": 1}})

        >>> mapping_transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseMappingTransformer', 'init': {'copy': False, 'verbose': False, 'return_native': True, 'mappings': {'a': {'x': 1}}, 'return_dtypes': {'a': 'Int64'}}, 'fit': {'is_fitted_': True}}

        ```

        """
        json_dict = super().to_json()

        # replace columns arg with mappings arg
        del json_dict["init"]["columns"]

        # make sure mappings dict is sorted for consistent repr
        mappings = _sort_nested_dict(self.mappings)

        return_dtypes = _sort_dict(self.return_dtypes)

        json_dict["init"]["mappings"] = mappings
        json_dict["init"]["return_dtypes"] = return_dtypes

        return json_dict

    def _infer_return_type(
        self,
        col: str,
    ) -> str:
        """Infer return_dtypes from provided mappings.

        Returns
        -------
            str:
                inferred dtype, e.g. 'Float64'

        Examples
        --------
        ```pycon
        >>> BaseMappingTransformer(mappings={"a": {"Y": 1, "N": 0}})._infer_return_type(col="a")
        'Int64'

        ```

        """
        dtype = str(pl.Series(self.mappings[col].values()).dtype)

        if dtype == "Null":
            dtype = "Float64"
            warnings.warn(
                f"{self.classname()}: Mappings with Null dtype have been provided, narwhals does not expose this dtype, so type will be assumed as Float64. Alternate types can be set using the return_dtypes argument.",
                stacklevel=2,
            )

        return dtype

    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Check mappings dict has been fitted.

        Parameters
        ----------
        X : DataFrame
            Data to apply mappings to.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Input X, copied if specified by user.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseMappingTransformer(
        ...     mappings={"a": {"Y": 1, "N": 0}},
        ...     return_dtypes={"a": "Int8"},
        ... )

        >>> test_df = pl.DataFrame({"a": ["Y", "N"], "b": [3, 4]})

        >>> # base class transform has no effect on data
        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ Y   ┆ 3   │
        │ N   ┆ 4   │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        return_native = self._process_return_native(return_native_override)

        self.check_is_fitted(["mappings", "return_dtypes", "is_fitted_"])

        X = super().transform(X, return_native_override=False)

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class BaseMappingTransformMixin(BaseTransformer):
    """Mixin class to apply mappings to columns method.

    Transformer uses the mappings attribute which should be a dict of dicts/mappings
    for each required column.

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

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = False

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Apply mapping defined in the mappings dict to each column in the columns attribute.

        Parameters
        ----------
        X : DataFrame
            Data with nominal columns to transform.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Transformed input X with levels mapped according to mappings dict.

        Raises
        ------
        TypeError:
            If columns to be mapped in X have unsupported type.

        #  not currently including doctest for this, as is not intended to be used
        #  independently (should be inherited as a mixin)

        """
        self.check_is_fitted(
            ["mappings", "return_dtypes", "mappings_from_null", "is_fitted_"]
        )

        X = _convert_dataframe_to_narwhals(X)

        backend = nw.get_native_namespace(X).__name__

        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        schema = X.collect_schema()

        if backend == "pandas":
            bool_input = any(
                isinstance(schema[col], nw.Boolean) for col in self.mappings
            )
            bool_output = "Boolean" in self.return_dtypes.values()
            if bool_input or bool_output:
                X = X.to_native()
                non_nullable_type_cols = [
                    col
                    for col in self.columns
                    if not isinstance(X[col].dtype, (*PANDAS_NULLABLE_TYPES,))
                ]
                if non_nullable_type_cols:
                    bad_types = {col: X[col].dtype for col in non_nullable_type_cols}
                    msg = f"{self.classname()}: For pandas, casting to boolean from non-nullable dtypes is unsafe, please use df.convert_dtypes to convert types of bad cols {bad_types}."
                    raise TypeError(msg)
                X = nw.from_native(X)

        mapping_exprs = []

        remaining_cols = self.columns

        for column_type, mapping_function in zip(
            [
                [nw.Boolean],
                [nw.Float64, nw.Float32],
                [
                    nw.Int8,
                    nw.Int16,
                    nw.Int32,
                    nw.Int64,
                    nw.UInt8,
                    nw.UInt16,
                    nw.UInt32,
                    nw.UInt64,
                ],
                [nw.String],
                [nw.Categorical],
            ],
            [
                map_boolean_columns,
                map_float_columns,
                map_integer_columns,
                map_string_columns,
                map_categorical_columns,
            ],
            strict=True,
        ):
            selected_cols = [
                col
                for col in remaining_cols
                if isinstance(schema[col], (*column_type,))
            ]

            selected_mapping_exprs = mapping_function(
                cols=selected_cols,
                mappings=_filter_column_dict(self.mappings, selected_cols),
                mappings_from_null=_filter_column_dict(
                    self.mappings_from_null, selected_cols
                ),
                return_dtypes=_filter_column_dict(self.return_dtypes, selected_cols),
                # warnings are not needed, as class guards against bad types
                verbose=False,
            )

            mapping_exprs = [*mapping_exprs, *selected_mapping_exprs]

            remaining_cols = list(set(remaining_cols).difference(set(selected_cols)))

        if len(remaining_cols) != 0:
            bad_types = {col: schema[col] for col in remaining_cols}
            msg = f"{self.classname()}: The following columns have types which are not covered by the existing mapping logic {bad_types}"

            raise (TypeError(msg))

        X = (
            X.with_columns(
                *mapping_exprs,
            )
            if mapping_exprs
            else X
        )

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class MappingTransformer(BaseMappingTransformer, BaseMappingTransformMixin):
    """Transformer to map values in columns to other values e.g. to merge two levels into one.

    Note, the MappingTransformer does not require 'self-mappings' to be defined i.e. if you want
    to map a value to itself, you can omit this value from the mappings rather than having to
    map it to itself.

    This transformer inherits from BaseMappingTransformMixin as well as the BaseMappingTransformer,
    BaseMappingTransformer performs standard checks, while BasemappingTransformMixin handles the
    actual logic.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtype: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns

    **kwargs
        Arbitrary keyword arguments passed onto BaseMappingTransformer.init method.

    Attributes
    ----------
    mappings : dict
        Dictionary of mappings for each column individually. The dict passed to mappings in
        init is set to the mappings attribute.

    mappings_from_null: dict[str, Any]
        dict storing what null values will be mapped to. Generally best to use an imputer,
        but this functionality is useful for inverting pipelines.

    return_dtypes: dict[str, RETURN_DTYPES]
        Dictionary of col:dtype for returned columns

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

    extractable_exprs: bool
        class attribute, indicates whether expressions for transformation can be
        extracted form fitted transformer using `get_transform_exprs` method

    Examples
    --------
    ```pycon
    >>> transformer = MappingTransformer(
    ...     mappings={"a": {"Y": 1, "N": 0}},
    ...     return_dtypes={"a": "Int8"},
    ... )
    >>> transformer
    MappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                       return_dtypes={'a': 'Int8'})

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = transformer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MappingTransformer', 'init': {'copy': False, 'verbose': False, 'return_native': True, 'mappings': {'a': {'N': 0, 'Y': 1}}, 'return_dtypes': {'a': 'Int8'}}, 'fit': {'is_fitted_': True}}

    >>> MappingTransformer.from_json(json_dump)
    MappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                       return_dtypes={'a': 'Int8'})

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    extractable_exprs = False

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the input data X according to the mappings in the mappings attribute dict.

        This method calls the BaseMappingTransformMixin.transform. Note, this transform method is
        different to some of the transform methods in the nominal module, even though they also
        use the BaseMappingTransformMixin.transform method. Here, if a value does not exist in
        the mapping it is unchanged.

        Parameters
        ----------
        X : DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : DataFrame
            Transformed input X with levels mapped according to mappings dict.

        Examples
        --------
        ``pycon
        >>> import polars as pl

        >>> transformer = MappingTransformer(
        ...   mappings={'a': {'Y': 1, 'N': 0}},
        ...   return_dtypes={"a":"Int8"},
        ...    )

        >>> test_df=pl.DataFrame({'a': ["Y", "N"], 'b': [3,4]})

        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 0   ┆ 4   │
        └─────┴─────┘

        ```

        """
        self.check_is_fitted("is_fitted_")
        X = _convert_dataframe_to_narwhals(X)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        X = BaseMappingTransformMixin.transform(
            self,
            X,
            return_native_override=False,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)
