"""This module contains a transformer that applies capping to numeric columns."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np

from tubular.mixins import WeightColumnMixin
from tubular.numeric import BaseNumericTransformer

if TYPE_CHECKING:
    from narwhals.typing import FrameT


class BaseCappingTransformer(BaseNumericTransformer, WeightColumnMixin):
    polars_compatible = True

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Base class for capping transformers, contains functionality shared across capping
        transformer classes.

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

        polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

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
        """Performs checks on a dictionary passed to.

        Parameters
        ----------
        capping_values_dict: dict of form {column_name: [lower_cap, upper_cap]}

        dict_name: 'capping_values' or 'quantiles'

        Returns
        ----------
        None

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

    @nw.narwhalify
    def fit(self, X: FrameT, y: None = None) -> BaseCappingTransformer:
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

        """
        if self.weights_column:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

        super().fit(X, y)

        self.quantile_capping_values = {}

        native_backend = nw.get_native_namespace(X)

        if self.quantiles is not None:
            for col in self.columns:
                if self.weights_column is None:
                    weights_column = "dummy_weights_column"
                    X = X.with_columns(
                        nw.new_series(
                            name="dummy_weights_column",
                            values=[1] * len(X),
                            backend=native_backend,
                        ),
                    )

                else:
                    weights_column = self.weights_column

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
        """Method to call the weighted_quantile method and prepare the outputs.

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

        values_col: str
            name of relevant values column in data

        weights_column: str
            name of relevant weight column in data

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        """
        if quantiles[0] is None:
            quantiles = np.array([quantiles[1]])

            results_no_none = self.weighted_quantile(
                X,
                quantiles,
                values_column=values_column,
                weights_column=weights_column,
            )

            results = [None] + results_no_none

        elif quantiles[1] is None:
            quantiles = np.array([quantiles[0]])

            results_no_none = self.weighted_quantile(
                X,
                quantiles,
                values_column=values_column,
                weights_column=weights_column,
            )

            results = results_no_none + [None]

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
        """Method to calculate weighted quantiles.

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

        values_col: str
            name of relevant values column in data

        weights_column: str
            name of relevant weight column in data

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        Examples
        --------
        >>> x = CappingTransformer(capping_values={"a": [2, 10]})
        >>> quantiles_to_compute = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [1, 1, 1], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [0, 1, 0], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [1, 1, 0], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3, 4, 5], sample_weight = [1, 1, 1, 1, 1], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3, 4, 5], sample_weight = [1, 0, 1, 0, 1], quantiles = [0, 0.5, 1.0])
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 2.0, 5.0]

        """
        quantiles = np.array(quantiles)

        nan_filter = ~(nw.col(values_column).is_null())
        X = X.filter(nan_filter)

        zero_weight_filter = ~(nw.col(weights_column) == 0)
        X = X.filter(zero_weight_filter)

        X = X.sort(by=values_column, descending=False)

        weighted_quantiles = X.select(
            (nw.col(weights_column).cum_sum()) / (nw.col(weights_column).sum()),
        )

        # TODO - once narwhals implements interpolate, replace this with nw
        # syntax
        weighted_quantiles = weighted_quantiles.get_column(weights_column).to_numpy()
        values = X.get_column(values_column).to_numpy()

        return list(np.interp(quantiles, weighted_quantiles, values))

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Apply capping to columns in X.

        If cap_value_max is set, any values above cap_value_max will be set to cap_value_max. If cap_value_min
        is set any values below cap_value_min will be set to cap_value_min. Only works or numeric columns.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply capping to.

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with min and max capping applied to the specified columns.

        """

        X = nw.from_native(super().transform(X))

        self.check_is_fitted(["_replacement_values"])

        dict_attrs = ["_replacement_values"]

        if self.quantiles:
            self.check_is_fitted(["quantile_capping_values"])

            capping_values_for_transform = self.quantile_capping_values

            dict_attrs = dict_attrs + ["quantile_capping_values"]

        else:
            capping_values_for_transform = self.capping_values

            dict_attrs = dict_attrs + ["capping_values"]

        for attr_name in dict_attrs:
            if getattr(self, attr_name) == {}:
                msg = f"{self.classname()}: {attr_name} attribute is an empty dict - perhaps the fit method has not been run yet"
                raise ValueError(msg)

        for col in self.columns:
            cap_value_min = capping_values_for_transform[col][0]
            cap_value_max = capping_values_for_transform[col][1]

            replacement_min = self._replacement_values[col][0]
            replacement_max = self._replacement_values[col][1]

            for cap_value, replacement_value, condition in zip(
                [cap_value_min, cap_value_max],
                [replacement_min, replacement_max],
                [nw.col(col) < cap_value_min, nw.col(col) > cap_value_max],
            ):
                if cap_value is not None:
                    X = X.with_columns(
                        nw.when(
                            condition,
                        )
                        .then(
                            replacement_value,
                        )
                        .otherwise(
                            nw.col(col),
                        )
                        # make sure type is preserved for single row,
                        # e.g. mapping single row to int could convert
                        # from float to int
                        # TODO - look into better ways to achieve this
                        .cast(
                            X.get_column(col).dtype,
                        )
                        .alias(col),
                    )

        return X


class CappingTransformer(BaseCappingTransformer):
    """Transformer to cap numeric values at both or either minimum and maximum values.

    For max capping any values above the cap value will be set to the cap. Similarly for min capping
    any values below the cap will be set to the cap. Only works for numeric columns.

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

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
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

        """
        super().fit(X, y)

        return self


class OutOfRangeNullTransformer(BaseCappingTransformer):
    """Transformer to set values outside of a range to null.

    This transformer sets the cut off values in the same way as
    the CappingTransformer. So either the user can specify them
    directly in the capping_values argument or they can be calculated
    in the fit method, if the user supplies the quantiles argument.

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
        Replacement values when capping is applied. This will contain nulls for each column.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
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
        """Method to set the _replacement_values to have all null values.

        Keeps the existing keys in the _replacement_values dict and sets all values (except None) in the lists to np.NaN. Any None
        values remain in place.
        """

        _replacement_values = {}

        for k, cap_values_list in capping_values.items():
            null_replacements_list = [
                None if replace_value is not None else False
                for replace_value in cap_values_list
            ]

            _replacement_values[k] = null_replacements_list

        return _replacement_values

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

        """
        super().fit(X=X, y=y)

        if self.weights_column:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

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

        return self
