"""Cntains transformers that apply capping to numeric columns."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Optional

import narwhals as nw
import numpy as np

from tubular.mixins import WeightColumnMixin
from tubular.numeric import BaseNumericTransformer

if TYPE_CHECKING:
    from narwhals.typing import FrameT
from beartype import beartype

from tubular._stats import _weighted_quantile_expr
from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
)
from tubular.types import DataFrame, Series


class BaseCappingTransformer(BaseNumericTransformer, WeightColumnMixin):
    """Base class for capping transformers, contains functionality shared across capping transformer classes.

    Attributes
    ----------
    capping_values : dict or None
        Capping values to apply to each column, capping_values argument.

    quantiles : dict or None
        Quantiles to set capping values at from input data. Will be empty after init, values
        populated when fit is run.

    quantile_capping_values : dict or None
        Capping values learned from quantiles (if provided) to apply to each column.

    weights_column : str or None
        weights_column argument.

    _replacement_values : dict
        Replacement values when capping is applied. Will be a copy of capping_values.

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

    FITS = True

    jsonable = False

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        capping_values : dict or None, default = None
            Dictionary of capping values to apply to each column. The keys in the dict should be the
            column names and each item in the dict should be a list of length 2. Items in the lists
            should be ints or floats or None. The first item in the list is the minimum capping value
            and the second item in the list is the maximum capping value. If None is supplied for
            either value then that capping will not take place for that particular column. Both items
            in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

        quantiles : dict or None, default = None
            Dictionary of quantiles in the range [0, 1] to set capping values at for each column.
            The keys in the dict should be the column names and each item in the dict should be a
            list of length 2. Items in the lists should be ints or floats or None. The first item in the
            list is the lower quantile and the second item is the upper quantile to set the capping
            value from. The fit method calculates the values quantile from the input data X. If None is
            supplied for either value then that capping will not take place for that particular column.
            Both items in the lists cannot be None. Either one of capping_values or quantiles must be
            supplied.

        weights_column : str or None, default = None
            Optional weights column argument that can be used in combination with quantiles. Not used
            if capping_values is supplied. Allows weighted quantiles to be calculated.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        Raises
        ------
            ValueError: if capping values/quantiles passed are invalid

        Examples
        --------
            >>> BaseCappingTransformer(
            ... capping_values={'a': [10, 20], 'b': [1,3]},
            ...    )
            BaseCappingTransformer(capping_values={'a': [10, 20], 'b': [1, 3]})

        """
        if capping_values is None and quantiles is None:
            msg = f"{self.classname()}: both capping_values and quantiles are None, either supply capping values in the capping_values argument or supply quantiles that can be learnt in the fit method"
            raise ValueError(msg)

        if capping_values is not None and quantiles is not None:
            msg = f"{self.classname()}: both capping_values and quantiles are not None, supply one or the other"
            raise ValueError(msg)

        if capping_values is not None:
            self.check_capping_values_dict(capping_values, "capping_values")

            super().__init__(columns=list(capping_values.keys()), **kwargs)

        if quantiles is not None:
            self.check_capping_values_dict(quantiles, "quantiles")

            for k, quantile_values in quantiles.items():
                for quantile_value in quantile_values:
                    if (quantile_value is not None) and (
                        quantile_value < 0 or quantile_value > 1
                    ):
                        msg = f"{self.classname()}: quantile values must be in the range [0, 1] but got {quantile_value} for key {k}"
                        raise ValueError(msg)

            super().__init__(columns=list(quantiles.keys()), **kwargs)

        self.quantiles = quantiles
        self.capping_values = capping_values
        WeightColumnMixin.check_and_set_weight(self, weights_column)

        if capping_values:
            self._replacement_values = copy.deepcopy(self.capping_values)

    def check_capping_values_dict(
        self,
        capping_values_dict: dict[str, list[int | float | None]],
        dict_name: str,
    ) -> None:
        """Check passed dictionary.

        Parameters
        ----------
        capping_values_dict: dict[str, float]
            dict of form {column_name: [lower_cap, upper_cap]}

        dict_name: str
            'capping_values' or 'quantiles'

        Raises
        ------
            TypeError: if arguments have incorrect type (being lazy here as beartype will soon replace)

            ValueError: if capping values are invalid, e.g. lower_cap>upper_cap

        Examples
        --------
            >>> transformer=BaseCappingTransformer(
            ... capping_values={'a': [10, 20], 'b': [1,3]},
            ...    )

            >>> transformer.check_capping_values_dict(transformer.capping_values, 'capping_values')

        """
        if type(capping_values_dict) is not dict:
            msg = f"{self.classname()}: {dict_name} should be dict of columns and capping values"
            raise TypeError(msg)

        for k, cap_values in capping_values_dict.items():
            if type(k) is not str:
                msg = f"{self.classname()}: all keys in {dict_name} should be str, but got {type(k)}"
                raise TypeError(msg)

            if type(cap_values) is not list:
                msg = f"{self.classname()}: each item in {dict_name} should be a list, but got {type(cap_values)} for key {k}"
                raise TypeError(msg)

            if len(cap_values) != 2:
                msg = f"{self.classname()}: each item in {dict_name} should be length 2, but got {len(cap_values)} for key {k}"
                raise ValueError(msg)

            for cap_value in cap_values:
                if cap_value is not None:
                    if type(cap_value) not in [int, float]:
                        msg = f"{self.classname()}: each item in {dict_name} lists must contain numeric values or None, got {type(cap_value)} for key {k}"
                        raise TypeError(msg)

                    if np.isnan(cap_value) or np.isinf(cap_value):
                        msg = f"{self.classname()}: item in {dict_name} lists contains numpy NaN or Inf values"
                        raise ValueError(msg)

            if all(cap_value is not None for cap_value in cap_values) and (
                cap_values[0] >= cap_values[1]
            ):
                msg = f"{self.classname()}: lower value is greater than or equal to upper value for key {k}"
                raise ValueError(msg)

            if all(cap_value is None for cap_value in cap_values):
                msg = f"{self.classname()}: both values are None for key {k}"
                raise ValueError(msg)

    @beartype
    @nw.narwhalify
    def fit(self, X: DataFrame, y: Optional[Series] = None) -> BaseCappingTransformer:
        """Learn capping values from input data X.

        Calculates the quantiles to cap at given the quantiles dictionary supplied
        when initialising the transformer. Saves learnt values in the capping_values
        attribute.

        Parameters
        ----------
        X : pd/pl/nw.DataFrame
            A dataframe with required columns to be capped.

        y : None
            Required for pipeline.

        Returns
        -------
            BaseCappingTransformer: fitted instance of class

        Examples
        --------
        >>> import polars as pl

        >>> transformer=BaseCappingTransformer(
        ... quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]},
        ...    )

        >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})
        >>> test_target=pl.Series(name='target', values=[5,6,7,8])

        >>> transformer.fit(test_df, test_target)
        BaseCappingTransformer(quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]})

        """
        super().fit(X, y)

        backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=backend.__name__,
                return_native=False,
            )
        WeightColumnMixin.check_weights_column(self, X, weights_column)

        self.quantile_capping_values = {}

        if self.quantiles is not None:
            for col in self.columns:
                cap_values = self.prepare_quantiles(
                    X,
                    self.quantiles[col],
                    values_column=col,
                    weights_column=weights_column,
                )

                self.quantile_capping_values[col] = cap_values

                self._replacement_values = copy.deepcopy(self.quantile_capping_values)

        else:
            warnings.warn(
                f"{self.classname()}: quantiles not set so no fitting done in CappingTransformer",
                stacklevel=2,
            )

        return self

    @nw.narwhalify
    def prepare_quantiles(
        self,
        X: FrameT,
        quantiles: list[float],
        values_column: str,
        weights_column: str,
    ) -> list[int | float]:
        """Call the weighted_quantile method and prepare the outputs.

        If there are no None values in the supplied quantiles then the outputs from weighted_quantile
        are returned as is. If there are then prepare_quantiles removes the None values before
        calling weighted_quantile and adds them back into the output, in the same position, after
        calling.

        Parameters
        ----------
        X : FrameT
            Dataframe with relevant columns to calculate quantiles from.

        quantiles : list[float]
            Weighted quantiles to calculate. Must all be between 0 and 1.

        values_column: str
            name of relevant values column in data

        weights_column: str
            name of relevant weight column in data

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        Examples
        --------
            >>> import polars as pl

            >>> x = BaseCappingTransformer(capping_values={"a": [2, 10]})

            >>> df=pl.DataFrame({'a':[1,2,3], 'weight': [1,1,1]})

            >>> quantiles_to_compute = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            >>> computed_quantiles = x.prepare_quantiles(X=df, values_column='a', weights_column='weight', quantiles = quantiles_to_compute)
            >>> [round(q, 1) for q in computed_quantiles]
            [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.2), np.float64(1.5), np.float64(1.8), np.float64(2.1), np.float64(2.4), np.float64(2.7), np.float64(3.0)]


        """
        if quantiles[0] is None:
            quantiles = np.array([quantiles[1]])

            results_no_none = self.weighted_quantile(
                X,
                quantiles,
                values_column=values_column,
                weights_column=weights_column,
            )

            results = [None, *results_no_none]

        elif quantiles[1] is None:
            quantiles = np.array([quantiles[0]])

            results_no_none = self.weighted_quantile(
                X,
                quantiles,
                values_column=values_column,
                weights_column=weights_column,
            )

            results = [*results_no_none, None]

        else:
            results = self.weighted_quantile(
                X,
                quantiles,
                values_column=values_column,
                weights_column=weights_column,
            )

        return results

    @nw.narwhalify
    def weighted_quantile(
        self,
        X: FrameT,
        quantiles: list[float],
        values_column: str,
        weights_column: str,
    ) -> list[int | float]:
        """Calculate weighted quantiles.

        This method is adapted from the "Completely vectorized numpy solution" answer from user
        Alleo (https://stackoverflow.com/users/498892/alleo) to the following stackoverflow question;
        https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy. This
        method is also licenced under the CC-BY-SA terms, as the original code sample posted
        to stackoverflow (pre February 1, 2016) was.

        Method is similar to numpy.percentile, but supports weights. Supplied quantiles should be
        in the range [0, 1]. Method calculates cumulative % of weight for each observation,
        then interpolates between these observations to calculate the desired quantiles. Null values
        in the observations (values) and 0 weight observations are filtered out before
        calculating.

        Parameters
        ----------
        X : FrameT
            Dataframe with relevant columns to calculate quantiles from.

        quantiles : None
            Weighted quantiles to calculate. Must all be between 0 and 1.

        values_column: str
            name of relevant values column in data

        weights_column: str
            name of relevant weight column in data

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        Examples
        --------
        >>> import polars as pl
        >>> x = CappingTransformer(capping_values={"a": [2, 10]})
        >>> df=pl.DataFrame({'a':[1,2,3], 'weight': [1,1,1]})
        >>> quantiles_to_compute = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> computed_quantiles = x.weighted_quantile(X=df, values_column='a', weights_column='weight', quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.2), np.float64(1.5), np.float64(1.8), np.float64(2.1), np.float64(2.4), np.float64(2.7), np.float64(3.0)]

        >>> df=pl.DataFrame({'a': [1,2,3], 'weight': [0,1,0]})
        >>> computed_quantiles = x.weighted_quantile(X=df, values_column='a', weights_column='weight', quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0)]

        >>> df=pl.DataFrame({'a':[1,2,3], 'weight': [1,1,0]})
        >>> computed_quantiles = x.weighted_quantile(X=df, values_column='a', weights_column='weight', quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.2), np.float64(1.4), np.float64(1.6), np.float64(1.8), np.float64(2.0)]

        >>> df=pl.DataFrame({'a':[1,2,3,4,5], 'weight': [1,1,1,1,1]})
        >>> computed_quantiles = x.weighted_quantile(X=df, values_column='a', weights_column='weight', quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.5), np.float64(2.0), np.float64(2.5), np.float64(3.0), np.float64(3.5), np.float64(4.0), np.float64(4.5), np.float64(5.0)]

        >>> df=pl.DataFrame({'a': [1,2,3,4,5], 'weight': [1,0,1,0,1]})
        >>> computed_quantiles = x.weighted_quantile(X=df, values_column='a', weights_column='weight', quantiles = [0, 0.5, 1.0])
        >>> [round(q, 1) for q in computed_quantiles]
        [np.float64(1.0), np.float64(2.0), np.float64(5.0)]

        """
        quantiles = np.array(quantiles)

        not_null_expr = ~(nw.col(values_column).is_null())
        nonzero_weight_expr = ~(nw.col(weights_column) == 0)
        combined_filter = not_null_expr & nonzero_weight_expr

        X = X.sort(by=values_column, descending=False)

        weights_expr = nw.col(weights_column).filter(combined_filter)
        values_expr = nw.col(values_column).filter(combined_filter)

        weighted_quantiles_expr = _weighted_quantile_expr(
            initial_weights_expr=weights_expr
        )
        results_dict = X.select(weighted_quantiles_expr, values_expr).to_dict()

        # TODO - once narwhals implements interpolate, replace this with nw
        # syntax
        weighted_quantiles = results_dict[weights_column].to_numpy()
        values = results_dict[values_column].to_numpy()

        return list(np.interp(quantiles, weighted_quantiles, values))

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Apply capping to columns in X.

        If cap_value_max is set, any values above cap_value_max will be set to cap_value_max. If cap_value_min
        is set any values below cap_value_min will be set to cap_value_min. Only works or numeric columns.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply capping to.

        return_native_override: Optional[bool]
            Option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with min and max capping applied to the specified columns.

        Raises
        ------
            ValueError: if method is quantile capping and fit has not been called

        Examples
        --------
            >>> import polars as pl

            >>> transformer=BaseCappingTransformer(
            ... capping_values={'a': [10, 20], 'b': [1,3]},
            ...    )

            >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

            >>> transformer.transform(test_df)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 10  ┆ 3   ┆ 1   │
            │ 15  ┆ 2   ┆ 2   │
            │ 18  ┆ 3   ┆ 3   │
            │ 20  ┆ 1   ┆ 4   │
            └─────┴─────┴─────┘

        """
        self.check_is_fitted(["_replacement_values"])

        X = _convert_dataframe_to_narwhals(X)

        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        dict_attrs = ["_replacement_values"]

        if self.quantiles:
            self.check_is_fitted(["quantile_capping_values"])

            capping_values_for_transform = self.quantile_capping_values

            dict_attrs = [*dict_attrs, "quantile_capping_values"]

        else:
            capping_values_for_transform = self.capping_values

            dict_attrs = [*dict_attrs, "capping_values"]

        for attr_name in dict_attrs:
            if getattr(self, attr_name) == {}:
                msg = f"{self.classname()}: {attr_name} attribute is an empty dict - perhaps the fit method has not been run yet"
                raise ValueError(msg)
        exprs = {}
        for col in self.columns:
            cap_value_min = capping_values_for_transform[col][0]
            cap_value_max = capping_values_for_transform[col][1]

            replacement_min = self._replacement_values[col][0]
            replacement_max = self._replacement_values[col][1]

            if cap_value_min is not None and cap_value_max is not None:
                col_expr = (
                    nw.when(nw.col(col) < cap_value_min)
                    .then(replacement_min)
                    .otherwise(
                        nw.when(nw.col(col) > cap_value_max)
                        .then(replacement_max)
                        .otherwise(nw.col(col)),
                    )
                )
            elif cap_value_min is not None:
                col_expr = (
                    nw.when(nw.col(col) < cap_value_min)
                    .then(replacement_min)
                    .otherwise(nw.col(col))
                )
            elif cap_value_max is not None:
                col_expr = (
                    nw.when(nw.col(col) > cap_value_max)
                    .then(replacement_max)
                    .otherwise(nw.col(col))
                )
            else:
                col_expr = nw.col(col)

            # make sure type is preserved for single row,
            # e.g. mapping single row to int could convert
            # from float to int
            # TODO - look into better ways to achieve this
            exprs[col] = col_expr.cast(
                X[col].dtype,
            ).alias(col)

        X = X.with_columns(**exprs)

        return _return_narwhals_or_native_dataframe(X, return_native)


class CappingTransformer(BaseCappingTransformer):
    """Transformer to cap numeric values at both or either minimum and maximum values.

    For max capping any values above the cap value will be set to the cap. Similarly for min capping
    any values below the cap will be set to the cap. Only works for numeric columns.

    Attributes:
    ----------
    capping_values : dict or None
        Capping values to apply to each column, capping_values argument.

    quantiles : dict or None
        Quantiles to set capping values at from input data. Will be empty after init, values
        populated when fit is run.

    quantile_capping_values : dict or None
        Capping values learned from quantiles (if provided) to apply to each column.

    weights_column : str or None
        weights_column argument.

    _replacement_values : dict
        Replacement values when capping is applied. Will be a copy of capping_values.

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
    -------
    >>> import polars as pl

    >>> transformer=CappingTransformer(
    ... capping_values={'a': [10, 20], 'b': [1,3]},
    ...    )

    >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

    >>> transformer.transform(test_df)
    shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 10  ┆ 3   ┆ 1   │
    │ 15  ┆ 2   ┆ 2   │
    │ 18  ┆ 3   ┆ 3   │
    │ 20  ┆ 1   ┆ 4   │
    └─────┴─────┴─────┘

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = True

    jsonable = False

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        capping_values : dict or None, default = None
            Dictionary of capping values to apply to each column. The keys in the dict should be the
            column names and each item in the dict should be a list of length 2. Items in the lists
            should be ints or floats or None. The first item in the list is the minimum capping value
            and the second item in the list is the maximum capping value. If None is supplied for
            either value then that capping will not take place for that particular column. Both items
            in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

        quantiles : dict or None, default = None
            Dictionary of quantiles in the range [0, 1] to set capping values at for each column.
            The keys in the dict should be the column names and each item in the dict should be a
            list of length 2. Items in the lists should be ints or floats or None. The first item in the
            list is the lower quantile and the second item is the upper quantile to set the capping
            value from. The fit method calculates the values quantile from the input data X. If None is
            supplied for either value then that capping will not take place for that particular column.
            Both items in the lists cannot be None. Either one of capping_values or quantiles must be
            supplied.

        weights_column : str or None, default = None
            Optional weights column argument that can be used in combination with quantiles. Not used
            if capping_values is supplied. Allows weighted quantiles to be calculated.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(capping_values, quantiles, weights_column, **kwargs)

    @nw.narwhalify
    def fit(self, X: FrameT, y: None = None) -> CappingTransformer:
        """Learn capping values from input data X.

        Calculates the quantiles to cap at given the quantiles dictionary supplied
        when initialising the transformer. Saves learnt values in the capping_values
        attribute.

        Parameters
        ----------
        X : pd/pl.DataFrame
            A dataframe with required columns to be capped.

        y : None
            Required for pipeline.

        Returns
        -------
            CappingTransformer: fitted instance of class

        Example
        -------
            >>> import polars as pl

            >>> transformer=CappingTransformer(
            ... quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]},
            ...    )

            >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

            >>> transformer.fit(test_df)
            CappingTransformer(quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]})

        """
        super().fit(X, y)

        return self


class OutOfRangeNullTransformer(BaseCappingTransformer):
    """Transformer to set values outside of a range to null.

    This transformer sets the cut off values in the same way as
    the CappingTransformer. So either the user can specify them
    directly in the capping_values argument or they can be calculated
    in the fit method, if the user supplies the quantiles argument.

    Attributes:
    ----------
    capping_values : dict or None
        Capping values to apply to each column, capping_values argument.

    quantiles : dict or None
        Quantiles to set capping values at from input data. Will be empty after init, values
        populated when fit is run.

    quantile_capping_values : dict or None
        Capping values learned from quantiles (if provided) to apply to each column.

    weights_column : str or None
        weights_column argument.

    _replacement_values : dict
        Replacement values when capping is applied. This will contain nulls for each column.

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
    -------
    >>> import polars as pl

    >>> transformer=OutOfRangeNullTransformer(
    ... capping_values={'a': [10, 20], 'b': [1,3]},
    ...    )
    >>> transformer
    OutOfRangeNullTransformer(capping_values={'a': [10, 20], 'b': [1, 3]})

    # transform method is inherited so also demo that here
    >>> test_df=pl.DataFrame()

    >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

    >>> transformer.transform(test_df)
    shape: (4, 3)
    ┌──────┬──────┬─────┐
    │ a    ┆ b    ┆ c   │
    │ ---  ┆ ---  ┆ --- │
    │ i64  ┆ i64  ┆ i64 │
    ╞══════╪══════╪═════╡
    │ null ┆ null ┆ 1   │
    │ 15   ┆ 2    ┆ 2   │
    │ 18   ┆ null ┆ 3   │
    │ null ┆ 1    ┆ 4   │
    └──────┴──────┴─────┘

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = True

    jsonable = False

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        capping_values : dict or None, default = None
            Dictionary of capping values to apply to each column. The keys in the dict should be the
            column names and each item in the dict should be a list of length 2. Items in the lists
            should be ints or floats or None. The first item in the list is the minimum capping value
            and the second item in the list is the maximum capping value. If None is supplied for
            either value then that capping will not take place for that particular column. Both items
            in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

        quantiles : dict or None, default = None
            Dictionary of quantiles to set capping values at for each column. The keys in the dict
            should be the column names and each item in the dict should be a list of length 2. Items
            in the lists should be ints or floats or None. The first item in the list is the lower
            quantile and the second item is the upper quantile to set the capping value from. The fit
            method calculates the values quantile from the input data X. If None is supplied for
            either value then that capping will not take place for that particular column. Both items
            in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

        weights_column : str or None, default = None
            Optional weights column argument that can be used in combination with quantiles. Not used
            if capping_values is supplied. Allows weighted quantiles to be calculated.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(
            capping_values=capping_values,
            quantiles=quantiles,
            weights_column=weights_column,
            **kwargs,
        )

        if capping_values:
            self._replacement_values = OutOfRangeNullTransformer.set_replacement_values(
                self.capping_values,
            )

    @staticmethod
    def set_replacement_values(capping_values: dict[str, list[float]]) -> None:
        """Set the _replacement_values to have all null values.

        Keeps the existing keys in the _replacement_values dict and sets all values (except None) in the lists to np.NaN. Any None
        values remain in place.

        Example:
        -------
        >>> import polars as pl

        >>> capping_values={"a": [0.1, 0.2], "b": [None, 10]}

        >>> OutOfRangeNullTransformer.set_replacement_values(capping_values)
        {'a': [None, None], 'b': [False, None]}

        """
        replacement_values = {}

        for k, cap_values_list in capping_values.items():
            null_replacements_list = [
                None if replace_value is not None else False
                for replace_value in cap_values_list
            ]

            replacement_values[k] = null_replacements_list

        return replacement_values

    @nw.narwhalify
    def fit(self, X: FrameT, y: None = None) -> OutOfRangeNullTransformer:
        """Learn capping values from input data X.

        Calculates the quantiles to cap at given the quantiles dictionary supplied
        when initialising the transformer. Saves learnt values in the capping_values
        attribute.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with required columns to be capped.

        y : None
            Required for pipeline.

        Returns
        -------
            OutOfRangeNullTransformer: fitted instance of class

        Example
        -------
            >>> import polars as pl

            >>> transformer=OutOfRangeNullTransformer(
            ... quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]},
            ...    )

            >>> test_df=pl.DataFrame({'a': [1,15,18,25], 'b': [6,2,7,1], 'c':[1,2,3,4]})

            >>> transformer.fit(test_df)
            OutOfRangeNullTransformer(quantiles={'a': [0.01, 0.99], 'b': [0.05, 0.95]})

        """
        super().fit(X=X, y=y)

        backend = nw.get_native_namespace(X)

        original_weights_column = self.weights_column
        weights_column = original_weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=backend.__name__,
                return_native=False,
            )
        WeightColumnMixin.check_weights_column(self, X, weights_column)

        # need to overwrite attr for fit method to work
        self.weights_column = weights_column

        if self.quantiles:
            BaseCappingTransformer.fit(self, X=X, y=y)

            self._replacement_values = OutOfRangeNullTransformer.set_replacement_values(
                self.quantile_capping_values,
            )

        else:
            warnings.warn(
                f"{self.classname()}: quantiles not set so no fitting done in OutOfRangeNullTransformer",
                stacklevel=2,
            )

        # restore attr
        self.weights_column = original_weights_column

        return self
