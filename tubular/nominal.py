"""This module contains transformers that apply encodings to nominal columns."""
from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
from beartype import beartype
from narwhals.dtypes import DType  # noqa: F401

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _return_narwhals_or_native_dataframe,
)
from tubular.base import BaseTransformer
from tubular.imputers import MeanImputer, MedianImputer
from tubular.mapping import BaseMappingTransformer, BaseMappingTransformMixin
from tubular.mixins import DropOriginalMixin, SeparatorColumnMixin, WeightColumnMixin
from tubular.types import DataFrame, ListOfStrs, Series

if TYPE_CHECKING:
    from narwhals.typing import FrameT


class BaseNominalTransformer(BaseTransformer):
    """
    Base Transformer extension for nominal transformers.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    FITS = False

    @beartype
    def check_mappable_rows(
        self,
        X: DataFrame,
        present_values: Optional[dict[str, set[Any]]] = None,
    ) -> None:
        """Method to check that all the rows to apply the transformer to are able to be
        mapped according to the values in the mappings dict.

        Parameters
        ----------
        X : DataFrame
            Data to apply nominal transformations to.

        present_values: Optional[dict[str, set[Any]]]
            optionally provide dictionary of values present in data by column. Avoided recalculating
            specifically for validation checks.

        Raises
        ------
        ValueError
            If any of the rows in a column (c) to be mapped, could not be mapped according to
            the mapping dict in mappings[c].

        """
        self.check_is_fitted(["mappings"])

        X = _convert_dataframe_to_narwhals(X)

        if present_values is None:
            present_values = {
                col: set(X.get_column(col).unique()) for col in self.columns
            }

        value_diffs = {
            col: set(present_values[col]).difference(set(self.mappings[col]))
            for col in self.columns
        }

        raise_error = any(len(value_diffs[col]) != 0 for col in self.columns)

        if raise_error:
            columns_with_unmappable_rows = [
                col for col in self.columns if len(value_diffs[col]) != 0
            ]
            msg = f"{self.classname()}: nulls would be introduced into columns {', '.join(columns_with_unmappable_rows)} from levels not present in mapping"
            raise ValueError(msg)

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
        present_values: Optional[dict[str, set[Any]]] = None,
    ) -> DataFrame:
        """Base nominal transformer transform method.  Checks that all the rows are able to be
        mapped according to the values in the mappings dict and calls the BaseTransformer transform method.

        Parameters
        ----------
        X : DataFrame
            Data to apply nominal transformations to.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        present_values: Optional[dict[str, set[Any]]]
            optionally provide dictionary of values present in data by column. Avoided recalculating
            specifically for validation checks.

        Returns
        -------
        X : DataFrame
            Input X.

        """

        return_native = self._process_return_native(return_native_override)

        # specify which class to prevent additional inheritance calls
        X = BaseTransformer.transform(self, X, return_native_override=False)

        self.check_mappable_rows(X, present_values)

        return _return_narwhals_or_native_dataframe(X, return_native)


class NominalToIntegerTransformer(BaseNominalTransformer, BaseMappingTransformMixin):
    """Transformer to convert columns containing nominal values into integer values.

    The nominal levels that are mapped to integers are not ordered in any way.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    start_encoding : int, default = 0
        Value to start the encoding from e.g. if start_encoding = 0 then the encoding would be
        {'A': 0, 'B': 1, 'C': 3} etc.. or if start_encoding = 5 then the same encoding would be
        {'A': 5, 'B': 6, 'C': 7}. Can be positive or negative.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    start_encoding : int
        Value to start the encoding / mapping of nominal to integer from.

    mappings : dict
        Created in fit. A dict of key (column names) value (mappings between levels and integers for given
        column) pairs.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        start_encoding: int = 0,
        **kwargs: dict[str, bool],
    ) -> None:
        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

        # this transformer shouldn't really be used with huge numbers of levels
        # so setup to use int8 type
        # if there are more levels than this, will get a type error
        self.return_dtypes = {c: "Int8" for c in self.columns}

        if not isinstance(start_encoding, int):
            msg = f"{self.classname()}: start_encoding should be an integer"
            raise ValueError(msg)

        self.start_encoding = start_encoding

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Creates mapping between nominal levels and integer values for categorical variables.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on, this sets the nominal levels that can be mapped.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """
        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}

        for c in self.columns:
            col_values = X[c].unique()

            self.mappings[c] = {
                k: i for i, k in enumerate(col_values, self.start_encoding)
            }

            # if more levels than int8 type can handle, then error
            if len(self.mappings[c]) > 127:
                msg = f"{self.classname()}: column {c} has too many levels to encode"
                raise ValueError(
                    msg,
                )

        # use BaseMappingTransformer init to process args
        # extract null_mappings from mappings etc
        base_mapping_transformer = BaseMappingTransformer(
            mappings=self.mappings,
            return_dtypes=self.return_dtypes,
        )

        self.mappings = base_mapping_transformer.mappings
        self.mappings_from_null = base_mapping_transformer.mappings_from_null
        self.return_dtypes = base_mapping_transformer.return_dtypes

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method to apply integer encoding stored in the mappings attribute to
        each column in the columns attribute.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped according to mappings dict.

        """

        X = super().transform(X)

        return BaseMappingTransformMixin.transform(self, X)


class GroupRareLevelsTransformer(BaseTransformer, WeightColumnMixin):

    """Transformer to group together rare levels of nominal variables into a new level,
    labelled 'rare' (by default).

    Rare levels are defined by a cut off percentage, which can either be based on the
    number of rows or sum of weights. Any levels below this cut off value will be
    grouped into the rare level.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    cut_off_percent : float, default = 0.01
        Cut off for the percent of rows or percent of weight for a level, levels below
        this value will be grouped.

    weights_column : None or str, default = None
        Name of weights column that should be used so cut_off_percent applies to sum of weights
        rather than number of rows.

    rare_level_name : any,default = 'rare'.
        Must be of the same type as columns.
        Label for the new 'rare' level.

    record_rare_levels : bool, default = False
        If True, an attribute called rare_levels_record_ will be added to the object. This will be a dict
        of key (column name) value (level from column considered rare according to cut_off_percent) pairs.
        Care should be taken if working with nominal variables with many levels as this could potentially
        result in many being stored in this attribute.

    unseen_levels_to_rare : bool, default = True
        If True, unseen levels in new data will be passed to rare, if set to false they will be left unchanged.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    cut_off_percent : float
        Cut off percentage (either in terms of number of rows or sum of weight) for a given
        nominal level to be considered rare.

    non_rare_levels : dict
        Created in fit. A dict of non-rare levels (i.e. levels with more than cut_off_percent weight or rows)
        that is used to identify rare levels in transform.

    rare_level_name : any
        Must be of the same type as columns.
        Label for the new nominal level that will be added to group together rare levels (as
        defined by cut_off_percent).

    record_rare_levels : bool
        Should the 'rare' levels that will be grouped together be recorded? If not they will be lost
        after the fit and the only information remaining will be the 'non'rare' levels.

    rare_levels_record_ : dict
        Only created (in fit) if record_rare_levels is True. This is dict containing a list of
        levels that were grouped into 'rare' for each column the transformer was applied to.

    weights_column : str
        Name of weights columns to use if cut_off_percent should be in terms of sum of weight
        not number of rows.

    unseen_levels_to_rare : bool
        If True, unseen levels in new data will be passed to rare, if set to false they will be left unchanged.

    training_data_levels : dict[set]
        Dictionary containing the set of values present in the training data for each column in self.columns. It
        will only exist in if unseen_levels_to_rare is set to False.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        cut_off_percent: float = 0.01,
        weights_column: str | None = None,
        rare_level_name: str | list[str] | None = "rare",
        record_rare_levels: bool = True,
        unseen_levels_to_rare: bool = True,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if not isinstance(cut_off_percent, float):
            msg = f"{self.classname()}: cut_off_percent must be a float"
            raise ValueError(msg)

        if not ((cut_off_percent > 0) & (cut_off_percent < 1)):
            msg = f"{self.classname()}: cut_off_percent must be > 0 and < 1"
            raise ValueError(msg)

        self.cut_off_percent = cut_off_percent

        WeightColumnMixin.check_and_set_weight(self, weights_column)

        if not isinstance(rare_level_name, str):
            msg = f"{self.classname()}: rare_level_name must be a str"
            raise ValueError(msg)

        self.rare_level_name = rare_level_name

        if not isinstance(record_rare_levels, bool):
            msg = f"{self.classname()}: record_rare_levels must be a bool"
            raise ValueError(msg)

        self.record_rare_levels = record_rare_levels

        if not isinstance(unseen_levels_to_rare, bool):
            msg = f"{self.classname()}: unseen_levels_to_rare must be a bool"
            raise ValueError(msg)

        self.unseen_levels_to_rare = unseen_levels_to_rare

    @beartype
    def _check_str_like_columns(self, schema: nw.Schema) -> None:
        """check that transformer being called on only str-like columns

        Parameters
        ----------
        schema: nw.Schema
            schema of input data

        """

        str_like_columns = [
            col
            for col in self.columns
            if schema[col] in {nw.String, nw.Categorical, nw.Object}
        ]

        non_str_like_columns = set(self.columns).difference(
            set(
                str_like_columns,
            ),
        )

        if len(non_str_like_columns) != 0:
            msg = f"{self.classname()}: transformer must run on str-like columns, but got non str-like {non_str_like_columns}"
            raise TypeError(msg)

    @beartype
    def _check_for_nulls(self, X: DataFrame) -> None:
        """check that transformer being called on only non-null columns.

        Note, found including nulls to be quite complicated due to:
        - categorical variables make use of NaN not None
        - pl/nw categorical variables do not allow categories to be edited,
        so adjusting requires converting to str as interim step
        - NaNs are converted to nan, introducing complications

        As this transformer is generally used post imputation, elected to remove null
        functionality.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to transform

        """

        X = _convert_dataframe_to_narwhals(X)

        null_check_expressions = {
            col: nw.col(col).is_null().sum().alias(col) for col in self.columns
        }

        null_counts = X.select(**null_check_expressions)

        columns_with_nulls = [
            col for col in self.columns if null_counts[col].item() > 0
        ]

        if columns_with_nulls:
            msg = f"{self.classname()}: transformer can only fit/apply on columns without nulls, columns {', '.join(columns_with_nulls)} need to be imputed first"
            raise ValueError(msg)

    @beartype
    def fit(
        self,
        X: DataFrame,
        y: Optional[Series] = None,
    ) -> GroupRareLevelsTransformer:
        """Records non-rare levels for categorical variables.

        When transform is called, only levels records in non_rare_levels during fit will remain
        unchanged - all other levels will be grouped. If record_rare_levels is True then the
        rare levels will also be recorded.

        The label for the rare levels must be of the same type as the columns.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to identify non-rare levels from.

        y : None or or nw.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """

        X = _convert_dataframe_to_narwhals(X)

        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

        schema = X.schema

        self._check_str_like_columns(schema)

        self._check_for_nulls(X)

        self.non_rare_levels = {}

        if self.record_rare_levels:
            self.rare_levels_record_ = {}

        native_backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=native_backend.__name__,
                return_native=False,
            )

        for c in self.columns:
            cols_w_sums = X.group_by(c, drop_null_keys=False).agg(
                nw.col(weights_column).sum(),
            )

            total_w = X.select(nw.col(weights_column).sum()).item()

            cols_w_percents = cols_w_sums.select(
                nw.col(weights_column) / total_w,
                nw.col(c),
            )

            self.non_rare_levels[c] = (
                cols_w_percents.filter(
                    nw.col(weights_column) >= self.cut_off_percent,
                )
                .get_column(c)
                .to_list()
            )

            self.non_rare_levels[c] = sorted(self.non_rare_levels[c], key=str)

            if self.record_rare_levels:
                self.rare_levels_record_[c] = (
                    cols_w_percents.filter(
                        nw.col(weights_column) < self.cut_off_percent,
                    )
                    .get_column(c)
                    .to_list()
                )

                self.rare_levels_record_[c] = sorted(
                    self.rare_levels_record_[c],
                    key=str,
                )

        if not self.unseen_levels_to_rare:
            self.training_data_levels = {}
            for c in self.columns:
                self.training_data_levels[c] = set(X.get_column(c).unique().to_list())

        return self

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Grouped rare levels together into a new 'rare' level.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to with catgeorical variables to apply rare level grouping to.

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with rare levels grouped for into a new rare level.

        """
        X = BaseTransformer.transform(self, X, return_native_override=False)
        X = _convert_dataframe_to_narwhals(X)

        schema = X.schema

        self._check_str_like_columns(schema)

        self._check_for_nulls(X)

        self.check_is_fitted(["non_rare_levels"])

        # copy non_rare_levels, as unseen values may be added, and transform should not
        # change the transformer state
        non_rare_levels = copy.deepcopy(self.non_rare_levels)

        if not self.unseen_levels_to_rare:
            for c in self.columns:
                unseen_vals = set(
                    X.get_column(c).unique().to_list(),
                ).difference(
                    set(self.training_data_levels[c]),
                )
                non_rare_levels[c].extend(unseen_vals)

        transform_expressions = {
            c: nw.col(c).cast(
                nw.String,
            )
            if schema[c] in [nw.Categorical, nw.Enum]
            else nw.col(c)
            for c in self.columns
        }

        transform_expressions = {
            c: (
                nw.when(transform_expressions[c].is_in(non_rare_levels[c]))
                .then(transform_expressions[c])
                .otherwise(nw.lit(self.rare_level_name))
            )
            for c in self.columns
        }

        transform_expressions = {
            c: transform_expressions[c].cast(schema[c])
            if (schema[c] in [nw.Categorical, nw.Enum])
            else transform_expressions[c]
            for c in self.columns
        }

        X = X.with_columns(**transform_expressions)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class MeanResponseTransformer(
    BaseNominalTransformer,
    WeightColumnMixin,
    BaseMappingTransformMixin,
    DropOriginalMixin,
):
    """Transformer to apply mean response encoding. This converts categorical variables to
    numeric by mapping levels to the mean response for that level.

    For a continuous or binary response the categorical columns specified will have values
    replaced with the mean response for each category.

    For an n > 1 level categorical response, up to n binary responses can be created, which in
    turn can then be used to encode each categorical column specified. This will generate up
    to n * len(columns) new columns, of with names of the form {column}_{response_level}. The
    original columns will be removed from the dataframe. This functionality is controlled using
    the 'level' parameter. Note that the above only works for a n > 1 level categorical response.
    Do not use 'level' parameter for a n = 1 level numerical response. In this case, use the standard
    mean response transformer without the 'level' parameter.

    If a categorical variable contains null values these will not be transformed.

    The same weights and prior are applied to each response level in the multi-level case.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    prior : int, default = 0
        Regularisation parameter, can be thought of roughly as the size a category should be in order for
        its statistics to be considered reliable (hence default value of 0 means no regularisation).

    level : str, list or None, default = None
        Parameter to control encoding against a multi-level categorical response. For a continuous or
        binary response, leave this as None. In the multi-level case, set to 'all' to encode against every
        response level or provide a list of response levels to encode against.

    unseen_level_handling : str("Mean", "Median", "Lowest" or "Highest) or int/float, default = None
        Parameter to control the logic for handling unseen levels of the categorical features to encode in
        data when using transform method. Default value of None will output error when attempting to use transform
        on data with unseen levels in categorical columns to encode. Set this parameter to one of the options above
        in order to encode unseen levels in each categorical column with the mean, median etc. of
        each column. One can also pass an arbitrary int/float value to use for encoding unseen levels.

    return_type: Literal['float32', 'float64']
        What type to cast return column as, consider exploring float32 to save memory. Defaults to float32.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    columns : str or list
        Categorical columns to encode in the input data.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    prior : int, default = 0
        Regularisation parameter, can be thought of roughly as the size a category should be in order for
        its statistics to be considered reliable (hence default value of 0 means no regularisation).

    level : str, int, float, list or None, default = None
        Parameter to control encoding against a multi-level categorical response. If None the response will be
        treated as binary or continous, if 'all' all response levels will be encoded against and if it is a list of
        levels then only the levels specified will be encoded against.

    response_levels : list
        Only created in the mutli-level case. Generated from level, list of all the response levels to encode against.

    mappings : dict
        Created in fit. A nested Dict of {column names : column specific mapping dictionary} pairs.  Column
        specific mapping dictionaries contain {initial value : mapped value} pairs.

    mapped_columns : list
        Only created in the multi-level case. A list of the new columns produced by encoded the columns in self.columns
        against multiple response levels, of the form {column}_{level}.

    transformer_dict : dict
        Only created in the mutli-level case. A dictionary of the form level : transformer containing the mean response
        transformers for each level to be encoded against.

    unseen_levels_encoding_dict: dict
        Dict containing the values (based on chosen unseen_level_handling) derived from the encoded columns to use when handling unseen levels in data passed to transform method.

    return_type: Literal['float32', 'float64']
        What type to cast return column as. Defaults to float32.

    cast_method: Literal[np.float32, np,float64]
        Store the casting method associated to return_type

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: Optional[Union[str, list[str]]] = None,
        weights_column: Optional[str] = None,
        prior: int = 0,
        level: Optional[Union[float, int, str, list]] = None,
        unseen_level_handling: Optional[
            Union[float, int, Literal["mean", "median", "min", "max"]]
        ] = None,
        return_type: Literal["Float32", "Float64"] = "Float32",
        drop_original: bool = True,
        **kwargs: bool,
    ) -> None:
        if not prior >= 0:
            msg = f"{self.classname()}: prior should be positive int"
            raise ValueError(msg)

        WeightColumnMixin.check_and_set_weight(self, weights_column)

        self.prior = prior
        self.unseen_level_handling = unseen_level_handling
        self.return_type = return_type
        DropOriginalMixin.set_drop_original_column(self, drop_original=drop_original)

        self.MULTI_LEVEL = False

        if level == "all" or (isinstance(level, list)):
            self.MULTI_LEVEL = True

        # if working with single level, put into list for easier handling
        elif isinstance(level, (str, int, float)):
            level = [level]
            self.MULTI_LEVEL = True

        self.level = level

        # self.cast_method is used to cast mapping values, so uses numpy types
        if return_type == "Float64":
            self.cast_method = np.float64
        else:
            self.cast_method = np.float32

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    @beartype
    def _prior_regularisation(
        self,
        weighted_response_sum_over_groups_exprs: dict[str, nw.Expr],
        weight_sum_over_groups_exprs: dict[str, nw.Expr],
        weights_column: str,
    ) -> dict[str, nw.Expr]:
        """Regularise encoding values by pushing encodings of infrequent categories towards the global mean.  If prior is zero this will return target_means unaltered.

        Parameters
        ----------


        Returns
        -------
        prior_exprs: dict[str, nw.Expr]
            dictionary of format col:prior expression for col

        """

        self.global_mean = {}

        # reuse mean imputer logic to calculate global mean
        global_mean_imputer = MeanImputer(
            columns=self.response_columns,
            weights_column=weights_column,
        )

        global_mean_exprs = global_mean_imputer._get_mean_calculation_expressions(
            global_mean_imputer.columns,
            weights_column,
        )

        return {
            encoded_column + "_mapped": (
                weighted_response_sum_over_groups_exprs[
                    f"{self.encoded_columns_to_columns[encoded_column]}_{self.encoded_columns_to_response_columns[encoded_column]}"
                ]
                + (
                    global_mean_exprs[
                        self.encoded_columns_to_response_columns[encoded_column]
                    ]
                    * nw.lit(self.prior)
                )
            )
            / (
                weight_sum_over_groups_exprs[
                    self.encoded_columns_to_columns[encoded_column]
                ]
                + self.prior
            ).cast(
                getattr(nw, self.return_type),
            )
            for encoded_column in self.encoded_columns
        }

    def _setup_fit_multi_level(
        self,
        y_vals: list[Union[int, float]],
        response_column: str,
    ) -> None:
        """setup attrs needed for fit, for multi level case

        Parameters
        ----------

        y_vals: list[Union[int, float]]
            y values present in data

        response_column: str
            name of response column

        """

        self.response_levels = self.level

        if self.level == "all":
            self.response_levels = y_vals

        elif any(level not in y_vals for level in self.level):
            msg = "Levels contains a level to encode against that is not present in the response."
            raise ValueError(msg)

        self.column_to_encoded_columns = {
            c: [c + "_" + str(level) for level in self.response_levels]
            for c in self.columns
        }

        self.encoded_columns_to_response_columns = {
            c + "_" + str(level): response_column + "_" + str(level)
            for c in self.columns
            for level in self.response_levels
        }

        self.response_columns = [
            response_column + "_" + level for level in self.response_levels
        ]

    def setup_fit_single_level(self, response_column: str) -> None:
        """setup attrs needed for fit, for non-multi level case

        Parameters
        ----------
        response_column: str
            name of response column

        """

        # arbitrary len 1 iterable so logic can be shared with multi level
        self.response_levels = ["SINGLE_LEVEL"]

        self.column_to_encoded_columns = {c: [c] for c in self.columns}

        self.encoded_columns_to_response_columns = {
            c: response_column for c in self.columns
        }

        self.response_columns = [
            response_column,
        ]

    @beartype
    def fit(self, X: DataFrame, y: Series) -> MeanResponseTransformer:
        """Identify mapping of categorical levels to mean response values.

        If the user specified the weights_column arg in when initialising the transformer
        the weighted mean response will be calculated using that column.

        In the multi-level case this method learns which response levels are present and
        are to be encoded against.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to with catgeorical variable columns to transform and also containing response_column
            column.

        y : pd/pl.Series
            Response variable or target.

        """

        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}
        self.unseen_levels_encoding_dict = {}

        backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                backend=backend.__name__,
                return_native=False,
            )

        WeightColumnMixin.check_weights_column(self, X, weights_column)

        y_vals = y.unique().to_list()

        response_null_count = sum(pd.isna(val) for val in y_vals)

        if response_null_count > 0:
            msg = f"{self.classname()}: y has {response_null_count} null values"
            raise ValueError(msg)

        X_y = self._combine_X_y(X, y, return_native_override=False)
        response_column = "_temporary_response"

        if self.MULTI_LEVEL:
            self._setup_fit_multi_level(y_vals, response_column)

        else:
            self.setup_fit_single_level(response_column)

        self.encoded_columns_to_columns = {
            encoded_column: c
            for c in self.columns
            for encoded_column in self.column_to_encoded_columns[c]
        }

        self.encoded_columns = [
            encoded_column
            for c in self.columns
            for encoded_column in self.column_to_encoded_columns[c]
        ]

        encoded_column_exprs = {
            encoded_column: nw.col(
                self.encoded_columns_to_columns[encoded_column],
            ).alias(encoded_column)
            for encoded_column in self.encoded_columns
        }

        response_exprs = {
            response_column + "_" + level if self.MULTI_LEVEL else response_column: (
                nw.col(response_column) == level
            )
            if self.MULTI_LEVEL
            else nw.col(response_column)
            for level in self.response_levels
        }

        weighted_response_exprs = {
            "weighted_" + response_column: response_exprs[response_column]
            * nw.col(weights_column)
            for response_column in self.response_columns
        }

        all_response_exprs = {}
        all_response_exprs.update(response_exprs)
        all_response_exprs.update(weighted_response_exprs)

        # need to materialise these here in order to use .over with pandas
        X_y = X_y.with_columns(**all_response_exprs)

        weighted_response_sum_over_groups_exprs = {
            f"{c}_{response_column}": nw.col("weighted_" + response_column)
            .sum()
            .over(c)
            for response_column in self.response_columns
            for c in self.columns
        }

        weight_sum_over_groups_exprs = {
            c: nw.col(weights_column).sum().over(c) for c in self.columns
        }

        prior_exprs = self._prior_regularisation(
            weighted_response_sum_over_groups_exprs,
            weight_sum_over_groups_exprs,
            weights_column,
        )

        full_expr_dict = {}
        full_expr_dict.update(prior_exprs)
        full_expr_dict.update(encoded_column_exprs)

        results_dict = X_y.select(**full_expr_dict).to_dict(as_series=True)

        self.mappings.update(
            {
                encoded_column: dict(
                    zip(
                        results_dict[encoded_column].to_list(),
                        results_dict[f"{encoded_column}_mapped"].to_list(),
                    ),
                )
                for encoded_column in self.encoded_columns
            },
        )

        # set this attr up for BaseMappingTransformerMixin
        # this is used to cast the narwhals mapping df, so uses narwhals types
        self.return_dtypes = {col: self.return_type for col in self.encoded_columns}

        # use BaseMappingTransformer init to process args
        # extract null_mappings from mappings etc
        base_mapping_transformer = BaseMappingTransformer(
            mappings=self.mappings,
            return_dtypes=self.return_dtypes,
        )

        self.mappings = base_mapping_transformer.mappings
        self.mappings_from_null = base_mapping_transformer.mappings_from_null
        self.return_dtypes = base_mapping_transformer.return_dtypes

        self._fit_unseen_level_handling_dict(X_y, encoded_column_exprs, weights_column)

        return self

    @beartype
    def _fit_unseen_level_handling_dict(
        self,
        X_y: DataFrame,
        encoded_column_exprs: dict[str, nw.Expr],
        weights_column: str,
    ) -> None:
        """Learn values for unseen levels to be mapped to, potential cases depend on unseen_level_handling attr:
        - if int/float value has been provided, this will cast to the appropriate type
        and be directly used
        - if median/mean/min/max, the appropriate weighted statistic is calculated on the mapped data, and
        cast to the appropriate type

        Parameters
        ----------
        X_y : pd/pl.DataFrame
            Data to with categorical variable columns to transform and also containing response_column
            column.

        encoded_column_exprs: dict[str, nw.Expr]
            dict of format str: expression for creating initial encoded columns. Needed for Median
            unseen level option which requires intermediate materialisations.

        weights_column : str
            name of weights column

        """

        if isinstance(self.unseen_level_handling, (int, float)):
            self.unseen_levels_encoding_dict.update(
                {
                    c: self.cast_method(
                        self.unseen_level_handling,
                    )
                    for c in self.encoded_columns
                },
            )

        elif isinstance(self.unseen_level_handling, str):
            unseen_level_exprs = {}

            present_values = {c: X_y.get_column(c).unique() for c in self.columns}

            schema = X_y.schema

            mapping_expressions = self._process_mapping_expressions(
                present_values=present_values,
                unseen_level_handling=None,
                schema=schema,
            )

            if self.unseen_level_handling in ["mean", "median"]:
                if self.unseen_level_handling == "mean":
                    # have to call this many times as  weights column varies with c
                    unseen_level_exprs.update(
                        MeanImputer._get_mean_calculation_expressions(
                            self.encoded_columns,
                            weights_column,
                            initial_columns_exprs=mapping_expressions,
                        ),
                    )

                # else, median
                else:
                    for c in self.encoded_columns:
                        X_temp = X_y.with_columns(**encoded_column_exprs)
                        X_temp = X_temp.sort(c).filter(~nw.col(c).is_null())

                        unseen_level_exprs.update(
                            {
                                c: MedianImputer._get_median_calculation_expression(
                                    c,
                                    weights_column,
                                    initial_column_expr=mapping_expressions[c],
                                ),
                            },
                        )

            # else, min/max
            else:
                unseen_level_exprs.update(
                    {
                        c: getattr(mapping_expressions[c], self.unseen_level_handling)()
                        for c in self.encoded_columns
                    },
                )

            unseen_level_results = X_y.select(**unseen_level_exprs).to_dict(
                as_series=True,
            )

            self.unseen_levels_encoding_dict = {
                c: unseen_level_results[c].item(0) for c in self.encoded_columns
            }

            self.unseen_levels_encoding_dict = {
                c: self.cast_method(
                    self.unseen_levels_encoding_dict[c],
                )
                for c in self.encoded_columns
            }

    def _process_mapping_expressions(
        self,
        present_values: dict[str, list[Any]],
        unseen_level_handling: bool,
        schema: nw.Schema,
    ) -> dict[str, nw.Expr]:
        """process learnt mappings into expressions for transforming data

        Parameters
        ----------
        present_levels: dict[str, list[Any]]
            dict of format column: levels present in column

        unseen_level_handling: bool
            indicates whether to include unseen level handling values in mappings

        schema: nw.Schema
            data schema

        Returns
        ----------
        dict[str, nw.Expr]:
            dict of format column:expression for mapping column

        """

        # set up list of paired condition/outcome tuples for mapping
        conditions_and_outcomes = {
            output_col: [
                self._create_mapping_conditions_and_outcomes(
                    input_col,
                    key,
                    self.mappings,
                    output_col=output_col,
                )
                if schema[input_col] not in [nw.Categorical, nw.Enum]
                else self._create_mapping_conditions_and_outcomes(
                    input_col,
                    key,
                    self.mappings,
                    dtype=nw.String,
                    output_col=output_col,
                )
                for key in self.mappings[output_col]
                if key in present_values[input_col]
            ]
            for input_col in self.columns
            for output_col in self.column_to_encoded_columns[input_col]
        }

        if unseen_level_handling:
            unseen_level_condition_and_outcomes = {
                output_col: (
                    ~nw.col(input_col).is_in(self.mappings[output_col].keys()),
                    (nw.lit(self.unseen_levels_encoding_dict[output_col])),
                )
                for input_col in self.columns
                for output_col in self.column_to_encoded_columns[input_col]
            }

            conditions_and_outcomes = {
                col: conditions_and_outcomes[col]
                + [unseen_level_condition_and_outcomes[col]]
                for col in self.encoded_columns
            }

        # apply mapping using functools reduce to build expression
        transform_expressions = {
            output_col: self._combine_mappings_into_expression(
                input_col,
                conditions_and_outcomes,
                output_col,
            )
            for input_col in self.columns
            for output_col in self.column_to_encoded_columns[input_col]
        }

        return {
            col: transform_expressions[col].cast(getattr(nw, self.return_dtypes[col]))
            for col in self.encoded_columns
        }

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform method to apply mean response encoding stored in the mappings attribute to
        each column in the columns attribute.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        N.B. In the mutli-level case, this method briefly overwrites the self.columns attribute, but sets
        it back to the original value at the end.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd/pl.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """

        self.check_is_fitted(["mappings", "return_dtypes"])

        X = _convert_dataframe_to_narwhals(X)

        present_values = {col: set(X.get_column(col).unique()) for col in self.columns}
        schema = X.schema

        # with columns created, can now run parent transforms
        if self.unseen_level_handling:
            # do not want to run check_mappable_rows in this case, as will not like unseen values
            self.check_is_fitted(["unseen_levels_encoding_dict"])

            # BaseTransformer.transform as we do not want to run check_mappable_rows in BaseNominalTransformer
            # (it causes complications with unseen levels and new cols, so run later)
            X = BaseTransformer.transform(self, X, return_native_override=False)

        else:
            # mappings might look like {'a_blue': {'a': 1, 'b': 2,...}}
            # what we want to check is whether the values of a are covered
            # by the mappings, so temp change the mappings dict to focus on
            # the original columns and set back to original value after
            original_mappings = self.mappings
            self.mappings = {
                col: self.mappings[self.column_to_encoded_columns[col][0]]
                for col in self.columns
            }
            X = super().transform(
                X,
                return_native_override=False,
                present_values=present_values,
            )
            self.mappings = original_mappings

        transform_expressions = self._process_mapping_expressions(
            present_values=present_values,
            unseen_level_handling=self.unseen_level_handling,
            schema=schema,
        )

        X = X.with_columns(
            **transform_expressions,
        )

        columns_to_drop = [
            col for col in self.columns if col not in self.encoded_columns
        ]

        X = DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            columns_to_drop,
            return_native=False,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class OrdinalEncoderTransformer(
    BaseNominalTransformer,
    BaseMappingTransformMixin,
    WeightColumnMixin,
):
    """Transformer to encode categorical variables into ascending rank-ordered integer values variables by mapping
    it's levels to the target-mean response for that level.
    Values will be sorted in ascending order only i.e. categorical level with lowest target mean response to
    be encoded as 1, the next highest value as 2 and so on.

    If a categorical variable contains null values these will not be transformed.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    weights_column : str or None
        Weights column to use when calculating the mean response.

    mappings : dict
        Created in fit. Dict of key (column names) value (mapping of categorical levels to numeric,
        ordinal encoded response values) pairs.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        WeightColumnMixin.check_and_set_weight(self, weights_column)

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

        # this transformer shouldn't really be used with huge numbers of levels
        # so setup to use int8 type
        # if there are more levels than this, will get a type error
        self.return_dtypes = {c: "Int8" for c in self.columns}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Identify mapping of categorical levels to rank-ordered integer values by target-mean in ascending order.

        If the user specified the weights_column arg in when initialising the transformer
        the weighted mean response will be calculated using that column.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform and response_column column
            specified when object was initialised.

        y : pd.Series
            Response column or target.

        """
        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}

        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

        response_null_count = y.isna().sum()

        if response_null_count > 0:
            msg = f"{self.classname()}: y has {response_null_count} null values"
            raise ValueError(msg)

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        for c in self.columns:
            if self.weights_column is None:
                # get the indexes of the sorted target mean-encoded dict
                _idx_target_mean = list(
                    X_y.groupby([c])[response_column]
                    .mean()
                    .sort_values(ascending=True, kind="mergesort")
                    .index,
                )

                # create a dictionary whose keys are the levels of the categorical variable
                # sorted ascending by their target-mean value
                # and whose values are ascending ordinal integers
                ordinal_encoded_dict = {
                    k: _idx_target_mean.index(k) + 1 for k in _idx_target_mean
                }

                self.mappings[c] = ordinal_encoded_dict

            else:
                groupby_sum = X_y.groupby([c])[
                    [response_column, self.weights_column]
                ].sum()

                # get the indexes of the sorted target mean-encoded dict
                _idx_target_mean = list(
                    (groupby_sum[response_column] / groupby_sum[self.weights_column])
                    .sort_values(ascending=True, kind="mergesort")
                    .index,
                )

                # create a dictionary whose keys are the levels of the categorical variable
                # sorted ascending by their target-mean value
                # and whose values are ascending ordinal integers
                ordinal_encoded_dict = {
                    k: _idx_target_mean.index(k) + 1 for k in _idx_target_mean
                }

                self.mappings[c] = ordinal_encoded_dict

        for col in self.columns:
            # if more levels than int8 type can handle, then error
            if len(self.mappings[col]) > 127:
                msg = f"{self.classname()}: column {c} has too many levels to encode"
                raise ValueError(
                    msg,
                )

        # use BaseMappingTransformer init to process args
        # extract null_mappings from mappings etc
        base_mapping_transformer = BaseMappingTransformer(
            mappings=self.mappings,
            return_dtypes=self.return_dtypes,
        )

        self.mappings = base_mapping_transformer.mappings
        self.mappings_from_null = base_mapping_transformer.mappings_from_null
        self.return_dtypes = base_mapping_transformer.return_dtypes

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method to apply ordinal encoding stored in the mappings attribute to
        each column in the columns attribute. This maps categorical levels to rank-ordered integer values by target-mean in ascending order.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed data with levels mapped to ordinal encoded values for categorical variables.

        """
        X = super().transform(X)

        return BaseMappingTransformMixin.transform(self, X)


class OneHotEncodingTransformer(
    DropOriginalMixin,
    SeparatorColumnMixin,
    BaseTransformer,
):
    """Transformer to convert categorical variables into dummy columns.

    Parameters
    ----------
    columns : str or list of strings or None, default = None
        Names of columns to transform. If the default of None is supplied all object and category
        columns in X are used.

    wanted_values: dict[str, list[str] or None , default = None
        Optional parameter to select specific column levels to be transformed. If it is None, all levels in the categorical column will be encoded. It will take the format {col1: [level_1, level_2, ...]}.

    separator : str
        Used to create dummy column names, the name will take
        the format [categorical feature][separator][category level]

    drop_original : bool, default = False
        Should original columns be dropped after creating dummy fields?

    copy : bool, default = False
        Should X be copied prior to transform? Should X be copied prior to transform? Copy argument no longer used and will be deprecated in a future release

    verbose : bool, default = True
        Should warnings/checkmarks get displayed?

    **kwargs
        Arbitrary keyword arguments passed onto sklearn OneHotEncoder.init method.

    Attributes
    ----------
    separator : str
        Separator used in naming for dummy columns.

    drop_original : bool
        Should original columns be dropped after creating dummy fields?

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: Optional[Union[str, ListOfStrs]] = None,
        wanted_values: Optional[dict[str, ListOfStrs]] = None,
        separator: str = "_",
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        **kwargs: bool,
    ) -> None:
        BaseTransformer.__init__(
            self,
            columns=columns,
            verbose=verbose,
            copy=copy,
            **kwargs,
        )

        self.wanted_values = wanted_values
        self.set_drop_original_column(drop_original)
        self.check_and_set_separator_column(separator)

    @beartype
    def _check_for_nulls(self, present_levels: dict[str, Any]) -> None:
        """check that transformer being called on only non-null columns.

        Note, found including nulls to be quite complicated due to:
        - categorical variables make use of NaN not None
        - pl/nw categorical variables do not allow categories to be edited,
        so adjusting requires converting to str as interim step
        - NaNs are converted to nan, introducing complications

        As this transformer is generally used post imputation, elected to remove null
        functionality.

        Parameters
        ----------
        present_levels: dict[str, Any]
            dict containing present levels per column

        """
        columns_with_nulls = []

        for c in present_levels:
            if any(pd.isna(val) for val in present_levels[c]):
                columns_with_nulls.append(c)

            if columns_with_nulls:
                msg = f"{self.classname()}: transformer can only fit/apply on columns without nulls, columns {', '.join(columns_with_nulls)} need to be imputed first"
                raise ValueError(msg)

    @beartype
    def fit(
        self,
        X: DataFrame,
        y: Optional[Series] = None,
    ) -> OneHotEncodingTransformer:
        """Gets list of levels for each column to be transformed. This defines which dummy columns
        will be created in transform.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to identify levels from.

        y : None
            Ignored. This parameter exists only for compatibility with sklearn.pipeline.Pipeline.

        """
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        BaseTransformer.fit(self, X=X, y=y)

        # first handle checks
        present_levels = {}
        for c in self.columns:
            # print warning for unseen levels
            present_levels[c] = set(X.get_column(c).unique().to_list())

        self._check_for_nulls(present_levels)

        # sort once nulls excluded
        present_levels = {c: sorted(present_levels[c]) for c in present_levels}

        self.categories_ = {}
        self.new_feature_names_ = {}
        # Check each field has less than 100 categories/levels
        missing_levels = {}
        for c in self.columns:
            level_count = len(present_levels[c])

            if level_count > 100:
                raise ValueError(
                    f"{self.classname()}: column %s has over 100 unique values - consider another type of encoding"
                    % c,
                )
            # categories if 'values' is provided
            final_categories = (
                present_levels[c]
                if self.wanted_values is None
                else self.wanted_values.get(c, None)
            )

            self.categories_[c] = final_categories
            self.new_feature_names_[c] = self._get_feature_names(column=c)

            missing_levels = self._warn_missing_levels(
                present_levels[c],
                c,
                missing_levels,
            )

        return self

    @beartype
    def _warn_missing_levels(
        self,
        present_levels: list[Any],
        c: str,
        missing_levels: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """Logs a warning for user-specifed levels that are not found in the dataset and updates "missing_levels[c]" with those missing levels.

        Parameters
        ----------
        present_levels: list
            List of levels observed in the data.
        c: str
            The column name being checked for missing user-specified levels.
        missing_levels: dict[str, list[str]]
            Dictionary containing missing user-specified levels for each column.
        Returns
        -------
        missing_levels : dict[str, list[str]]
            Dictionary updated to reflect new missing levels for column c

        """
        # print warning for missing levels
        missing_levels[c] = sorted(
            set(self.categories_[c]).difference(set(present_levels)),
        )
        if len(missing_levels[c]) > 0:
            warning_msg = f"{self.classname()}: column {c} includes user-specified values {missing_levels[c]} not found in the dataset"
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

        return missing_levels

    @beartype
    def _get_feature_names(
        self,
        column: str,
    ) -> list[str]:
        """Function to get list of features that will be output by transformer

        Parameters
        ----------
        column: str
            column to get dummy feature names for

        """

        return [
            column + self.separator + str(level) for level in self.categories_[column]
        ]

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Create new dummy columns from categorical fields.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply one hot encoding to.

        return_native_override: Optional[bool]
        option to override return_native attr in transformer, useful when calling parent
        methods

        Returns
        -------
        X_transformed : pd/pl.DataFrame
            Transformed input X with dummy columns derived from categorical columns added. If drop_original
            = True then the original categorical columns that the dummies are created from will not be in
            the output X.

        """
        return_native = self._process_return_native(return_native_override)

        # Check that transformer has been fit before calling transform
        self.check_is_fitted(["categories_"])

        X = _convert_dataframe_to_narwhals(X)
        X = BaseTransformer.transform(self, X, return_native_override=False)

        # first handle checks
        present_levels = {}
        for c in self.columns:
            # print warning for unseen levels
            present_levels[c] = set(X.get_column(c).unique().to_list())
            unseen_levels = set(present_levels[c]).difference(set(self.categories_[c]))
            if len(unseen_levels) > 0:
                warning_msg = f"{self.classname()}: column {c} has unseen categories: {unseen_levels}"
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

        self._check_for_nulls(present_levels)

        # sort once nulls excluded
        present_levels = {key: sorted(present_levels[key]) for key in present_levels}

        missing_levels = {}
        transform_expressions = {}
        for c in self.columns:
            # print warning for missing levels
            missing_levels = self._warn_missing_levels(
                present_levels[c],
                c,
                missing_levels,
            )

            wanted_dummies = self.new_feature_names_[c]

            for level in present_levels[c]:
                if c + self.separator + str(level) in wanted_dummies:
                    transform_expressions[c + self.separator + str(level)] = (
                        nw.col(c) == level
                    )

            for level in missing_levels[c]:
                transform_expressions[c + self.separator + str(level)] = nw.lit(
                    False,
                ).alias(c + self.separator + str(level))

        # make column order consistent
        sorted_keys = sorted(transform_expressions.keys())

        X = X.with_columns(**{key: transform_expressions[key] for key in sorted_keys})

        # Drop original columns if self.drop_original is True
        X = DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )

        return _return_narwhals_or_native_dataframe(X, return_native)
