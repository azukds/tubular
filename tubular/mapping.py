"""This module contains transformers that apply different types of mappings to columns."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
from beartype import beartype
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer
from tubular.types import DataFrame

if TYPE_CHECKING:
    from narwhals.typing import IntoDType


class BaseMappingTransformer(BaseTransformer):
    """Base Transformer Extension for mapping transformers.

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
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

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

    Example:
    --------
    >>> BaseMappingTransformer(
    ...   mappings={'a': {'Y': 1, 'N': 0}},
    ...   return_dtypes={"a":"Int8"},
    ...    )
    BaseMappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                           return_dtypes={'a': 'Int8'})

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = True

    RETURN_DTYPES = Literal[
        "String",
        "Object",
        "Categorical",
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Float32",
        "Float64",
    ]

    @beartype
    def __init__(
        self,
        mappings: dict[str, dict[Any, Any]],
        return_dtypes: Union[dict[str, RETURN_DTYPES], None] = None,
        **kwargs: Optional[bool],
    ) -> None:
        if not len(mappings) > 0:
            msg = f"{self.classname()}: mappings has no values"
            raise ValueError(msg)

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
            return_dtypes[col] = self._infer_return_type(mappings, col)

        self.return_dtypes = return_dtypes

        super().__init__(columns=columns, **kwargs)

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """dump transformer to json dict

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        >>> mapping_transformer=BaseMappingTransformer(mappings={'a': {'x': 1}})

        >>> mapping_transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseMappingTransformer', 'init': {'copy': False, 'verbose': False, 'return_native': True, 'mappings': {'a': {'x': 1}}, 'return_dtypes': {'a': 'Int64'}}, 'fit': {}}
        """

        json_dict = super().to_json()

        # replace columns arg with mappings arg
        del json_dict["init"]["columns"]
        json_dict["init"]["mappings"] = self.mappings
        json_dict["init"]["return_dtypes"] = self.return_dtypes

        return json_dict

    @staticmethod
    def _infer_return_type(
        mappings: dict[str, dict[str, str | float | int]],
        col: str,
    ) -> str:
        """infer return_dtypes from provided mappings

        Example:
        --------
        >>> BaseMappingTransformer._infer_return_type({"a": {"Y": 1, "N":0}}, col="a")
        'Int64'
        """

        return str(pl.Series(mappings[col].values()).dtype)

    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Base mapping transformer transform method.  Checks that the mappings
        dict has been fitted and calls the BaseTransformer transform method.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply mappings to.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : pd/pl.DataFrame
            Input X, copied if specified by user.

        Example:
        --------
        >>> import polars as pl

        >>> transformer = BaseMappingTransformer(
        ...   mappings={'a': {'Y': 1, 'N': 0}},
        ...   return_dtypes={"a":"Int8"},
        ...    )

        >>> test_df=pl.DataFrame({'a': ["Y", "N"], 'b': [3,4]})

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
        """

        X = _convert_dataframe_to_narwhals(X)

        return_native = self._process_return_native(return_native_override)

        self.check_is_fitted(["mappings", "return_dtypes"])

        X = super().transform(X, return_native_override=False)

        return _return_narwhals_or_native_dataframe(X, return_native)


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

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    @staticmethod
    def _create_mapping_conditions_and_outcomes(
        input_col: str,
        key: str,
        mappings: dict[str, dict[str, Union[int, str, bool, float]]],
        dtype: Optional[IntoDType] = None,
        output_col: Optional[str] = None,
    ) -> tuple[nw.Expr, nw.Expr]:
        """Applies the mapping defined in the mappings dict to each column in the columns
        attribute.

        Parameters
        ----------
        input_col : str
            column to be mapped

        key: str
            mapping key (value in column) to prepare condition/outcome pair for

        mappings: dict[str, dict[str,Union[int, str, bool, float]]]
            mappings  for column

        dtype: Optional[nw.IntoDType]
            dtype for values being mapped to. Generally narwhals will just infer this, but
            has some issues with categorical variables so can be necessary to cast to string
            for these (and then cast back after mapping).

        output_col : Optional[str]
            name of output column, which will be present in mappings.
            If none then defaults to input_col.

        Returns
        -------
        Tuple[nw.Expr, nw.Expr]: prepared pair of mapping condition/outcome

        # currently not including doctests for this, as need to look into most meaningful
        # way to doctest functions which output expressions
        """
        if output_col is None:
            output_col = input_col

        return (
            (
                nw.col(input_col) == key,
                nw.lit(mappings[output_col][key]),
            )
            if dtype is None
            else (
                nw.col(input_col) == key,
                nw.lit(mappings[output_col][key], dtype=dtype),
            )
        )

    @staticmethod
    def _combine_mappings_into_expression(
        input_col: str,
        conditions_and_outcomes: dict[str, tuple[nw.Expr, nw.Expr]],
        output_col: Optional[str] = None,
    ) -> nw.Expr:
        """combines mapping conditions/outcomes into one expr for given column

        Parameters
        ----------
        input_col : str
            column to prepare mappings for

        conditions_and_outcomes: List[Tuple[nw.Expr, nw.Expr]]
            list of paired conditions/outcomes to be used in mapping expression

        output_col : Optional[str]
            name of column to output in expression. If none then defaults to input_col.

        Returns
        -------
        nw.Expr: prepared mapping expression

        # currently not including doctests for this, as need to look into most meaningful
        # way to doctest functions which output expressions

        """

        if output_col is None:
            output_col = input_col

        if len(conditions_and_outcomes[output_col]) == 0:
            return nw.col(input_col)

        initial_condition = (
            nw.when(conditions_and_outcomes[output_col][0][0])
            .then(conditions_and_outcomes[output_col][0][1])
            .otherwise(nw.col(input_col))
            .alias(output_col)
        )

        if len(conditions_and_outcomes[output_col]) > 1:
            # chain together list of conditions/outcomes
            # e.g. [(condition1, outcome1), (condition2, outcome2)]
            # nw.when(condition2).then(outcome2).otherwise(
            # nw.when(condition1).then(outcome1).otherwise(nw.col(col))
            # )
            return reduce(
                lambda expr, condition_and_outcome: nw.when(condition_and_outcome[0])
                .then(condition_and_outcome[1])
                .otherwise(expr),
                conditions_and_outcomes[output_col][
                    1:
                ],  # start reduce logic after first entry
                initial_condition,
            )

        # if only one condition, just return this
        return initial_condition

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Applies the mapping defined in the mappings dict to each column in the columns
        attribute.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data with nominal columns to transform.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        #  not currently including doctest for this, as is not intended to be used
        #  independently (should be inherited as a mixin)

        """
        self.check_is_fitted(["mappings", "return_dtypes", "mappings_from_null"])

        X = _convert_dataframe_to_narwhals(X)

        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        # if the column is categorical, narwhals struggles to infer a type
        # during the when/then logic, so we need to record this so that we can
        # tell polars to use string as a common type
        # types are then corrected before returning at the end
        schema = X.schema
        column_is_categorical = {
            col: bool(schema[col] in [nw.Categorical, nw.Enum]) for col in self.mappings
        }

        present_values = {col: set(X.get_column(col).unique()) for col in self.mappings}

        # set up list of paired condition/outcome tuples for mapping
        conditions_and_outcomes = {
            col: [
                self._create_mapping_conditions_and_outcomes(col, key, self.mappings)
                if not column_is_categorical[col]
                else self._create_mapping_conditions_and_outcomes(
                    col,
                    key,
                    self.mappings,
                    nw.String,
                )
                for key in self.mappings[col]
                # nulls handled separately with fill_null call
                if ((not pd.isna(key)) and (key in present_values[col]))
            ]
            for col in self.mappings
        }

        # apply mapping using functools reduce to build expression
        transform_expressions = {
            col: self._combine_mappings_into_expression(col, conditions_and_outcomes)
            for col in self.mappings
        }

        # finally, handle mappings from null (imputations)
        transform_expressions = {
            col: (transform_expressions[col].fill_null(self.mappings_from_null[col]))
            if self.mappings_from_null[col] is not None
            else transform_expressions[col]
            for col in transform_expressions
        }

        # handle casting for non-bool return types
        # (bool has special handling at end)
        transform_expressions = {
            col: transform_expressions[col].cast(getattr(nw, self.return_dtypes[col]))
            if self.return_dtypes[col] != "Boolean"
            else transform_expressions[col]
            for col in transform_expressions
        }

        X = X.with_columns(
            **transform_expressions,
        )

        # this last section is needed to ensure pandas bool columns
        # are returned in sensible (non object) types
        # maybe_convert_dtypes will not run on an expression,
        # so do need a second with_columns call
        if "Boolean" in self.return_dtypes.values():
            X = X.with_columns(
                nw.maybe_convert_dtypes(X[col]).cast(
                    getattr(nw, self.return_dtypes[col]),
                )
                if self.return_dtypes[col] == "Boolean"
                else nw.col(col)
                for col in self.mappings
            )

        return _return_narwhals_or_native_dataframe(X, return_native)


class MappingTransformer(BaseMappingTransformer, BaseMappingTransformMixin):
    """Transformer to map values in columns to other values e.g. to merge two levels into one.

    Note, the MappingTransformer does not require 'self-mappings' to be defined i.e. if you want
    to map a value to itself, you can omit this value from the mappings rather than having to
    map it to itself. This is because it uses the pandas replace method which only replaces values
    which have a corresponding mapping.

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

    Example:
    --------
    >>> transformer = MappingTransformer(
    ...   mappings={'a': {'Y': 1, 'N': 0}},
    ...   return_dtypes={"a":"Int8"},
    ...    )
    >>> transformer
    MappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                       return_dtypes={'a': 'Int8'})

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump=transformer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MappingTransformer', 'init': {'copy': False, 'verbose': False, 'return_native': True, 'mappings': {'a': {'Y': 1, 'N': 0}}, 'return_dtypes': {'a': 'Int8'}}, 'fit': {}}

    >>> MappingTransformer.from_json(json_dump)
    MappingTransformer(mappings={'a': {'N': 0, 'Y': 1}},
                       return_dtypes={'a': 'Int8'})

    """

    polars_compatible = True

    lazyframe_compatible = False

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
        X : pd/pl.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        Example:
        --------
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

        """

        X = _convert_dataframe_to_narwhals(X)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        mapped_columns = self.mappings.keys()

        col_values_present = {}
        for col in mapped_columns:
            values_to_be_mapped = set(self.mappings[col].keys())
            col_values_present[col] = set(X.get_column(col).unique())

            if self.verbose:
                if len(values_to_be_mapped.intersection(col_values_present[col])) == 0:
                    warnings.warn(
                        f"{self.classname()}: No values from mapping for {col} exist in dataframe.",
                        stacklevel=2,
                    )

                if len(values_to_be_mapped.difference(col_values_present[col])) > 0:
                    warnings.warn(
                        f"{self.classname()}: There are values in the mapping for {col} that are not present in the dataframe",
                        stacklevel=2,
                    )

        X = BaseMappingTransformMixin.transform(
            self,
            X,
            return_native_override=False,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class BaseCrossColumnMappingTransformer(BaseMappingTransformer):
    """BaseMappingTransformer Extension for cross column mapping transformers.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.

    mappings : dict or OrderedDict
        Dictionary containing adjustments. Exact structure will vary by child class.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of mappings for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.

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

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    def __init__(
        self,
        adjust_column: str,
        mappings: dict[str, dict],
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(mappings=mappings, **kwargs)

        if not isinstance(adjust_column, str):
            msg = f"{self.classname()}: adjust_column should be a string"
            raise TypeError(msg)

        self.adjust_column = adjust_column

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Checks X is valid for transform and calls parent transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        X = super().transform(X)

        if self.adjust_column not in X.columns.to_numpy():
            msg = f"{self.classname()}: variable {self.adjust_column} is not in X"
            raise ValueError(msg)

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class CrossColumnMappingTransformer(BaseCrossColumnMappingTransformer):
    """Transformer to adjust values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.

    mappings : dict or OrderedDict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 'a', 3: 'b'}, 'b': {'a': 1, 'b': 2}}
        would replace the values in the adjustment column based off the values in column a using the mapping
        1->'a', 3->'b' and also replace based off the values in column b using a mapping 'a'->1, 'b'->2.
        If more than one column is defined for this mapping, then this object must be an OrderedDict
        to ensure reproducibility.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of mappings for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.

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

    lazyframe_compatible = False

    jsonable = False

    FITS = False

    def __init__(
        self,
        adjust_column: str,
        mappings: dict[str, dict],
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(mappings=mappings, adjust_column=adjust_column, **kwargs)

        if len(mappings) > 1 and not isinstance(mappings, OrderedDict):
            msg = f"{self.classname()}: mappings should be an ordered dict for 'replace' mappings using multiple columns"
            raise TypeError(msg)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        X = super().transform(X)

        for i in self.columns:
            for j in self.mappings[i]:
                X[self.adjust_column] = np.where(
                    (X[i] == j),
                    self.mappings[i][j],
                    X[self.adjust_column],
                )

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class BaseCrossColumnNumericTransformer(BaseCrossColumnMappingTransformer):
    """BaseCrossColumnNumericTransformer Extension for cross column numerical mapping transformers.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.

    mappings : dict
        Dictionary containing adjustments. Exact structure will vary by child class.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of mappings for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.

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

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    def __init__(
        self,
        adjust_column: str,
        mappings: dict[str, dict],
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(mappings=mappings, adjust_column=adjust_column, **kwargs)

        for j in mappings.values():
            for k in j.values():
                if type(k) not in [int, float]:
                    msg = f"{self.classname()}: mapping values must be numeric"
                    raise TypeError(msg)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Checks X is valid for transform and calls parent transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        X = super().transform(X)

        if not pd.api.types.is_numeric_dtype(X[self.adjust_column]):
            msg = f"{self.classname()}: variable {self.adjust_column} must have numeric dtype."
            raise TypeError(msg)

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class CrossColumnMultiplyTransformer(BaseCrossColumnNumericTransformer):
    """Transformer to apply a multiplicative adjustment to values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.  The data type of this column must be int or float.

    mappings : dict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 5}, 'b': {'a': 0.5, 'b': 1.1}}
        would multiply the values in the adjustment column based off the values in column a using the mapping
        1->2*value, 3->5*value and also multiply based off the values in column b using a mapping
        'a'->0.5*value, 'b'->1.1*value.
        The values within the dicts defining the multipliers must have type int or float.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of multiplicative adjustments for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.

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

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    def __init__(
        self,
        adjust_column: str,
        mappings: dict[str, dict],
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(mappings=mappings, adjust_column=adjust_column, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        X = super().transform(X)

        for i in self.columns:
            for j in self.mappings[i]:
                X[self.adjust_column] = np.where(
                    (X[i] == j),
                    X[self.adjust_column] * self.mappings[i][j],
                    X[self.adjust_column],
                )

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class CrossColumnAddTransformer(BaseCrossColumnNumericTransformer):
    """Transformer to apply an additive adjustment to values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.  The data type of this column must be int or float.

    mappings : dict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 5}, 'b': {'a': 1, 'b': -5}}
        would provide an additive adjustment to the values in the adjustment column based off the values
        in column a using the mapping 1->2+value, 3->5+value and also an additive adjustment based off the
        values in column b using a mapping 'a'->1+value, 'b'->(-5)+value.
        The values within the dicts defining the values to be added must have type int or float.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of additive adjustments for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.

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

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    def __init__(
        self,
        adjust_column: str,
        mappings: dict[str, dict],
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(mappings=mappings, adjust_column=adjust_column, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        X = super().transform(X)

        for i in self.columns:
            for j in self.mappings[i]:
                X[self.adjust_column] = np.where(
                    (X[i] == j),
                    X[self.adjust_column] + self.mappings[i][j],
                    X[self.adjust_column],
                )

        return X
