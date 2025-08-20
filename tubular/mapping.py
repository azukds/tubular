"""This module contains transformers that apply different types of mappings to columns."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
from beartype import beartype

from tubular._utils import new_narwhals_series_with_optimal_pandas_types
from tubular.base import BaseTransformer
from tubular.types import DataFrame

if TYPE_CHECKING:
    from narwhals.typing import FrameT


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

    return_dtypes: dict[str, RETURN_DTYPES]
        Dictionary of col:dtype for returned columns

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

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

        null_mappings = {col: None for col in mappings}
        for col in mappings:
            null_keys = [key for key in mappings[col] if pd.isna(key)]

            if len(null_keys) > 1:
                multi_null_map_msg = f"Multiple mappings have been provided for null values in column {col}, transformer is set up to handle nan/None/NA as one"
                raise ValueError(
                    multi_null_map_msg,
                )

            # Assign the mapping to the single null key if it exists
            if len(null_keys) != 0:
                null_mappings[col] = mappings[col][null_keys[0]]

        self.mappings = mappings
        self.null_mappings = null_mappings

        columns = list(mappings.keys())

        # if return_dtypes is not provided, then infer from mappings
        if not return_dtypes:
            return_dtypes = self._infer_return_types(mappings)

        self.return_dtypes = return_dtypes

        super().__init__(columns=columns, **kwargs)

    @staticmethod
    def _infer_return_types(
        mappings: dict[str, dict[str, str | float | int]],
    ) -> dict[str, str]:
        "infer return_dtypes from provided mappings"

        return {col: str(pl.Series(mappings[col].values()).dtype) for col in mappings}

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Base mapping transformer transform method.  Checks that the mappings
        dict has been fitted and calls the BaseTransformer transform method.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : pd/pl.DataFrame
            Input X, copied if specified by user.

        """
        self.check_is_fitted(["mappings", "return_dtypes"])

        return super().transform(X)


class BaseMappingTransformMixin(BaseTransformer):
    """Mixin class to apply mappings to columns method.

    Transformer uses the mappings attribute which should be a dict of dicts/mappings
    for each required column.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    @beartype
    @nw.narwhalify
    def transform(self, X: DataFrame) -> DataFrame:
        """Applies the mapping defined in the mappings dict to each column in the columns
        attribute.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """
        self.check_is_fitted(["mappings", "return_dtypes", "null_mappings"])

        X = nw.from_native(super().transform(X))
        native_backend = nw.get_native_namespace(X)

        # will do a join further down, which does not preserve index
        # polars does not care about this, but pandas does,
        # so need to handle a bit carefully
        if native_backend.__name__ == "pandas":
            index = nw.to_native(X).index

        # pull out column order to preserve
        column_order = X.columns

        for col in self.mappings:
            mappings = self.mappings[col]

            # TODO - update this logic once narwhals implements map_dict
            # differentiate between unmapped cols and cols mapped to null
            # by including unmapped cols mapped to self
            unique = X.get_column(col).unique()
            mappings = {
                key: mappings.get(key, key)
                for key in unique
                # nulls are handled separately at the end,
                # treated as imputations
                if not pd.isna(key)
            }
            new_col_values = f"new_{col}_values"

            # have to be careful at each stage to avoid pandas
            # default non-nullable types
            dtypes = {}
            dtypes["keys"] = X.get_column(col).dtype
            dtypes["values"] = getattr(nw, self.return_dtypes[col])

            # from_dict also appears to struggle with category dtypes
            # so convert to string and back again later
            temp_dtypes = {}
            for dtype_key in ["keys", "values"]:
                temp_dtypes[dtype_key] = dtypes[dtype_key]
                if dtypes[dtype_key] == nw.Categorical:
                    temp_dtypes[dtype_key] = nw.String

            mappings_keys = new_narwhals_series_with_optimal_pandas_types(
                name=col,
                values=list(mappings.keys()),
                backend=native_backend.__name__,
                dtype=temp_dtypes["keys"],
            )

            mappings_values = new_narwhals_series_with_optimal_pandas_types(
                name=new_col_values,
                values=list(mappings.values()),
                backend=native_backend.__name__,
                dtype=temp_dtypes["values"],
            )

            mappings_df = nw.from_dict(
                {
                    col: mappings_keys,
                    new_col_values: mappings_values,
                },
                backend=native_backend,
            )

            # bring back category types, and keep forcing better bool type
            for dtype_key, col_name in zip(["keys", "values"], [col, new_col_values]):
                if dtypes[dtype_key] == nw.Categorical:
                    mappings_df = mappings_df.with_columns(
                        nw.col(col_name).cast(nw.Categorical),
                    )

                if self.return_dtypes[col] == "Boolean":
                    mappings_df = mappings_df.with_columns(
                        nw.maybe_convert_dtypes(mappings_df[col_name]),
                    )

            X = (
                X.join(
                    mappings_df,
                    how="left",
                    on=col,
                )
                .drop(col)
                .rename({new_col_values: col})
            )

            # null keys are not joined on, so just fill nulls at end
            if self.null_mappings[col] is not None:
                X = X.with_columns(nw.col(col).fill_null(self.null_mappings[col]))
                X = X.with_columns(
                    nw.col(col).cast(getattr(nw, self.return_dtypes[col])),
                )

        # restore original index for pandas
        if nw.get_native_namespace(X).__name__ == "pandas":
            X = nw.to_native(X)
            X.index = index

        return X[column_order]


class MappingTransformer(BaseMappingTransformer, BaseMappingTransformMixin):
    """Transformer to map values in columns to other values e.g. to merge two levels into one.

    Note, the MappingTransformer does not require 'self-mappings' to be defined i.e. if you want
    to map a value to itself, you can omit this value from the mappings rather than having to
    map it to itself. This is because it uses the pandas replace method which only replaces values
    which have a corresponding mapping.

    This transformer inherits from BaseMappingTransformMixin as well as the BaseMappingTransformer
    in order to access the pd.Series.replace transform function.

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

    return_dtypes: dict[str, RETURN_DTYPES]
        Dictionary of col:dtype for returned columns

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    @nw.narwhalify
    def transform(
        self,
        X: FrameT,
    ) -> FrameT:
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

        """

        BaseTransformer.transform(self, X)

        mapped_columns = self.mappings.keys()

        for col in mapped_columns:
            values_to_be_mapped = set(self.mappings[col].keys())
            values_in_df = set(X.get_column(col).unique())

            if self.verbose:
                if len(values_to_be_mapped.intersection(values_in_df)) == 0:
                    warnings.warn(
                        f"{self.classname()}: No values from mapping for {col} exist in dataframe.",
                        stacklevel=2,
                    )

                if len(values_to_be_mapped.difference(values_in_df)) > 0:
                    warnings.warn(
                        f"{self.classname()}: There are values in the mapping for {col} that are not present in the dataframe",
                        stacklevel=2,
                    )

        return nw.from_native(BaseMappingTransformMixin.transform(self, X))


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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework


    """

    polars_compatible = False

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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework


    """

    polars_compatible = False

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
