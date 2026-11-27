"""Contains transformers that apply different types of mappings to columns."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl
from beartype import beartype
from typing_extensions import deprecated

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

# single source of truth for which input column types are handled by which
# mapping function. Used by BaseMappingTransformMixin (dispatches across all
# types) and the type specific transformers (each use one entry).
MAPPING_TYPE_REGISTRY = {
    "boolean": ((nw.Boolean,), map_boolean_columns),
    "float": ((nw.Float64, nw.Float32), map_float_columns),
    "integer": (
        (
            nw.Int8,
            nw.Int16,
            nw.Int32,
            nw.Int64,
            nw.UInt8,
            nw.UInt16,
            nw.UInt32,
            nw.UInt64,
        ),
        map_integer_columns,
    ),
    "string": ((nw.String,), map_string_columns),
    "categorical": ((nw.Categorical,), map_categorical_columns),
}


def _check_pandas_boolean_safety(
    X: nw.DataFrame,
    columns: list[str],
    return_dtypes: dict[str, str],
    bool_input: bool,
    classname: str,
) -> None:
    """Check pandas columns are nullable types if booleans are involved in mapping.

    Casting to/from boolean from non-nullable pandas dtypes is unsafe, as it
    may corrupt null values. No-op for non-pandas backends.

    Parameters
    ----------
    X: nw.DataFrame
        data to be mapped

    columns: list[str]
        columns being mapped

    return_dtypes: dict[str, str]
        col:dtype for returned columns

    bool_input: bool
        whether any of the columns being mapped are boolean

    classname: str
        name of calling transformer, for error message

    Raises
    ------
    TypeError:
        if booleans are involved and any column has a non-nullable pandas dtype

    """
    if nw.get_native_namespace(X).__name__ != "pandas":
        return

    if not (bool_input or "Boolean" in return_dtypes.values()):
        return

    native_X = X.to_native()
    bad_types = {
        col: native_X[col].dtype
        for col in columns
        if not isinstance(native_X[col].dtype, (*PANDAS_NULLABLE_TYPES,))
    }
    if bad_types:
        msg = f"{classname}: For pandas, casting to boolean from non-nullable dtypes is unsafe, please use df.convert_dtypes to convert types of bad cols {bad_types}."
        raise TypeError(msg)


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
class _BaseTypedMappingTransformer(BaseMappingTransformer):
    """Base class for mapping transformers which only handle a single input column type.

    Subclasses set the class attributes below to specify which input column types
    are supported, and which function from tubular.functions.mapping builds the
    mapping expressions.

    Once the deprecated MappingTransformer is removed from the repo, I think it would
    make sense to condense this class into BaseMappingTransformer.

    Attributes
    ----------
    column_type: str
        class attribute, key into MAPPING_TYPE_REGISTRY, which defines the supported
        input dtypes and the tubular.functions.mapping function used to build expressions

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

    jsonable = True

    column_type: str

    @property
    def allowed_types(self) -> tuple[type, ...]:
        """Narwhals dtypes supported as input by this transformer."""
        return MAPPING_TYPE_REGISTRY[self.column_type][0]

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        mapping_function = MAPPING_TYPE_REGISTRY[self.column_type][1]
        return (
            *mapping_function(
                cols=self.columns,
                mappings=self.mappings,
                mappings_from_null=self.mappings_from_null,
                return_dtypes=self.return_dtypes,
                # warnings are not needed, as classes guard against bad types
                verbose=False,
            ),
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Apply mappings to the columns in the columns attribute.

        Parameters
        ----------
        X : DataFrame
            Data with columns to transform.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Transformed input X with values mapped according to mappings dict.

        Raises
        ------
        TypeError:
            If columns to be mapped in X have unsupported type.

        """
        self.check_is_fitted(
            ["mappings", "return_dtypes", "mappings_from_null", "is_fitted_"]
        )

        X = _convert_dataframe_to_narwhals(X)

        return_native = self._process_return_native(return_native_override)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        schema = X.collect_schema()

        bad_types = {
            col: schema[col]
            for col in self.columns
            if not isinstance(schema[col], self.allowed_types)
        }

        if bad_types:
            allowed_types_str = "/".join(t.__name__ for t in self.allowed_types)
            msg = f"{self.classname()}: transformer can only handle {allowed_types_str} type columns but got columns with types {bad_types}"
            raise TypeError(msg)

        _check_pandas_boolean_safety(
            X,
            columns=self.columns,
            return_dtypes=self.return_dtypes,
            bool_input=self.column_type == "boolean",
            classname=self.classname(),
        )

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class MapStringColumnsTransformer(_BaseTypedMappingTransformer):
    """Transformer to map values in string columns to other values.

    Only string type columns are supported, any other type will raise a TypeError
    in transform. If a value does not exist in the mapping it is unchanged (where
    the output type allows this). For mapping across mixed column types, use MappingTransformer.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns. If not provided, inferred from mappings.

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

    Examples
    --------
    ```pycon
    >>> import polars as pl

    >>> transformer = MapStringColumnsTransformer(
    ...     mappings={"a": {"Y": 1, "N": 0}},
    ...     return_dtypes={"a": "Int8"},
    ... )

    >>> test_df = pl.DataFrame({"a": ["Y", "N"]})

    >>> transformer.transform(test_df)
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i8  │
    ╞═════╡
    │ 1   │
    │ 0   │
    └─────┘

    ```

    """

    column_type = "string"


@register
class MapIntegerColumnsTransformer(_BaseTypedMappingTransformer):
    """Transformer to map values in integer columns to other values.

    Only integer type columns are supported, any other type will raise a TypeError
    in transform. If a value does not exist in the mapping it is unchanged (where
    the output type allows this). For mapping across mixed column types, use MappingTransformer.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns. If not provided, inferred from mappings.

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

    Examples
    --------
    ```pycon
    >>> import polars as pl

    >>> transformer = MapIntegerColumnsTransformer(
    ...     mappings={"a": {1: 10, 2: 20}},
    ...     return_dtypes={"a": "Int8"},
    ... )

    >>> test_df = pl.DataFrame({"a": [1, 2]})

    >>> transformer.transform(test_df)
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i8  │
    ╞═════╡
    │ 10  │
    │ 20  │
    └─────┘

    ```

    """

    column_type = "integer"


@register
class MapFloatColumnsTransformer(_BaseTypedMappingTransformer):
    """Transformer to map values in float columns to other values.

    Only float type columns are supported, any other type will raise a TypeError
    in transform. If a value does not exist in the mapping it is unchanged (where
    the output type allows this). For mapping across mixed column types, use MappingTransformer.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns. If not provided, inferred from mappings.

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

    Examples
    --------
    ```pycon
    >>> import polars as pl

    >>> transformer = MapFloatColumnsTransformer(
    ...     mappings={"a": {1.5: 10.0, 2.5: 20.0}},
    ...     return_dtypes={"a": "Float64"},
    ... )

    >>> test_df = pl.DataFrame({"a": [1.5, 2.5]})

    >>> transformer.transform(test_df)
    shape: (2, 1)
    ┌──────┐
    │ a    │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 10.0 │
    │ 20.0 │
    └──────┘

    ```

    """

    column_type = "float"


@register
class MapBooleanColumnsTransformer(_BaseTypedMappingTransformer):
    """Transformer to map values in boolean columns to other values.

    Only boolean type columns are supported, any other type will raise a TypeError
    in transform. If a value does not exist in the mapping it is unchanged (where
    the output type allows this). For mapping across mixed column types, use MappingTransformer.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns. If not provided, inferred from mappings.

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

    Examples
    --------
    ```pycon
    >>> import polars as pl

    >>> transformer = MapBooleanColumnsTransformer(
    ...     mappings={"a": {True: 1, False: 0}},
    ...     return_dtypes={"a": "Int8"},
    ... )

    >>> test_df = pl.DataFrame({"a": [True, False]})

    >>> transformer.transform(test_df)
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i8  │
    ╞═════╡
    │ 1   │
    │ 0   │
    └─────┘

    ```

    """

    column_type = "boolean"


@register
class MapCategoricalColumnsTransformer(_BaseTypedMappingTransformer):
    """Transformer to map values in categorical columns to other values.

    Only categorical type columns are supported, any other type will raise a TypeError
    in transform. If a value does not exist in the mapping it is unchanged (where
    the output type allows this). For mapping across mixed column types, use MappingTransformer.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    return_dtypes: Optional[Dict[str, RETURN_DTYPES]]
        Dictionary of col:dtype for returned columns. If not provided, inferred from mappings.

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

    Examples
    --------
    ```pycon
    >>> import polars as pl

    >>> transformer = MapCategoricalColumnsTransformer(
    ...     mappings={"a": {"Y": 1, "N": 0}},
    ...     return_dtypes={"a": "Int8"},
    ... )

    >>> test_df = pl.DataFrame(
    ...     {"a": ["Y", "N"]}, schema={"a": pl.Categorical}
    ... )

    >>> transformer.transform(test_df)
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i8  │
    ╞═════╡
    │ 1   │
    │ 0   │
    └─────┘

    ```

    """

    column_type = "categorical"


# DEPRECATED CLASSES
@register
@deprecated(
    """This mixin has been deprecated, and will be removed in a future version.
    """,
)
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

        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        schema = X.collect_schema()

        _check_pandas_boolean_safety(
            X,
            columns=self.columns,
            return_dtypes=self.return_dtypes,
            bool_input=any(
                isinstance(schema[col], nw.Boolean) for col in self.mappings
            ),
            classname=self.classname(),
        )

        mapping_exprs = []

        remaining_cols = self.columns

        for column_types, mapping_function in MAPPING_TYPE_REGISTRY.values():
            selected_cols = [
                col for col in remaining_cols if isinstance(schema[col], column_types)
            ]

            mapping_exprs = [
                *mapping_exprs,
                *mapping_function(
                    cols=selected_cols,
                    mappings=_filter_column_dict(self.mappings, selected_cols),
                    mappings_from_null=_filter_column_dict(
                        self.mappings_from_null, selected_cols
                    ),
                    return_dtypes=_filter_column_dict(
                        self.return_dtypes, selected_cols
                    ),
                    # warnings are not needed, as classes guard against bad types
                    verbose=False,
                ),
            ]

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
@deprecated(
    """This transformer has been deprecated and replaced with type specific transformer
    classes, please use MapBooleanColumnsTransformer/MapCategoricalColumnsTransformer/
    MapFloatColumnsTransformer/MapIntegerColumnsTransformer/MapStringColumnsTransformer
    from this file instead.
    """,
)
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
