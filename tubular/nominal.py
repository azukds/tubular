"""This module contains transformers that apply encodings to nominal columns."""
from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Literal, Optional, Union

import narwhals as nw
import narwhals.selectors as ncs
import numpy as np
from beartype import beartype

from tubular.base import BaseTransformer
from tubular.imputers import MeanImputer, MedianImputer
from tubular.mapping import BaseMappingTransformer, BaseMappingTransformMixin
from tubular.mixins import DropOriginalMixin, SeparatorColumnMixin, WeightColumnMixin

if TYPE_CHECKING:
    import pandas as pd
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

    @nw.narwhalify
    def check_mappable_rows(self, X: FrameT) -> None:
        """Method to check that all the rows to apply the transformer to are able to be
        mapped according to the values in the mappings dict.

        Raises
        ------
        ValueError
            If any of the rows in a column (c) to be mapped, could not be mapped according to
            the mapping dict in mappings[c].

        """
        self.check_is_fitted(["mappings"])

        for c in self.columns:
            mappable_rows = X.select(
                nw.col(c).is_in(list(self.mappings[c])).sum(),
            ).item()

            if mappable_rows < X.shape[0]:
                msg = f"{self.classname()}: nulls would be introduced into column {c} from levels not present in mapping"
                raise ValueError(msg)

    @nw.narwhalify
    def transform(self, X: FrameT) -> None:
        """Base nominal transformer transform method.  Checks that all the rows are able to be
        mapped according to the values in the mappings dict and calls the BaseTransformer transform method.

        Parameters
        ----------
        X : FrameT
            Data to apply nominal transformations to.

        Returns
        -------
        X : FrameT
            Input X.

        """

        # specify which class to prevent additional inheritance calls
        X = BaseTransformer.transform(self, X)

        self.check_mappable_rows(X)

        return X


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
        self.null_mappings = base_mapping_transformer.null_mappings
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

    @nw.narwhalify
    def _check_str_like_columns(self, X: FrameT) -> None:
        """check that transformer being called on only str-like columns

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to transform

        """

        str_like_columns = list(
            set(self.columns).intersection(
                set(
                    X.select(
                        ncs.string(),
                        ncs.categorical(),
                        ncs.by_dtype(nw.Object),
                    ).columns,
                ),
            ),
        )

        non_str_like_columns = set(self.columns).difference(
            set(
                str_like_columns,
            ),
        )

        if len(non_str_like_columns) != 0:
            msg = f"{self.classname()}: transformer must run on str-like columns, but got non str-like {non_str_like_columns}"
            raise TypeError(msg)

    @nw.narwhalify
    def _check_for_nulls(self, X: FrameT) -> None:
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

        for c in self.columns:
            nulls_count = 0

            # pick out str_like dtypes that allow nans
            nulls_count += X.select(nw.col(c).is_null().sum()).item()

            if nulls_count != 0:
                msg = f"{self.classname()}: transformer can only fit/apply on columns without nulls, column {c} needs to be imputed first"
                raise ValueError(msg)

    @nw.narwhalify
    def fit(self, X: FrameT, y: nw.Series | None = None) -> FrameT:
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
        super().fit(X, y)

        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

        self._check_str_like_columns(
            X.with_columns(nw.col(col) for col in self.columns),
        )

        self._check_for_nulls(X)

        self.non_rare_levels = {}

        if self.record_rare_levels:
            self.rare_levels_record_ = {}

        native_backend = nw.get_native_namespace(X)

        weights_column = self.weights_column
        if self.weights_column is None:
            weights_column = "dummy_weights_column"
            X = X.with_columns(
                nw.new_series(
                    name=weights_column,
                    values=[1] * len(X),
                    backend=native_backend,
                ),
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

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
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
        X = nw.from_native(BaseTransformer.transform(self, X))

        self._check_str_like_columns(
            X.with_columns(nw.col(col) for col in self.columns),
        )

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
                for unseen_val in unseen_vals:
                    non_rare_levels[c].append(unseen_val)

        for c in self.columns:
            categorical = False
            if str(X.schema[c]) == "Categorical":
                categorical = True
                X = X.with_columns(nw.col(c).cast(nw.String))

            non_rare_condition = nw.col(c).is_in(non_rare_levels[c])

            X = X.with_columns(
                nw.when(non_rare_condition)
                .then(
                    nw.col(c),
                )
                .otherwise(
                    nw.lit(self.rare_level_name),
                )
                .alias(
                    c,
                ),
            )

            if categorical:
                X = X.with_columns(nw.col(c).cast(nw.Categorical))

        return X


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

    @nw.narwhalify
    def _prior_regularisation(
        self,
        group_means_and_weights: FrameT,
        column: str,
        weights_column: str,
        response_column: str,
    ) -> dict[str | float, float]:
        """Regularise encoding values by pushing encodings of infrequent categories towards the global mean.  If prior is zero this will return target_means unaltered.

        Parameters
        ----------
        group_means_and_weights: FrameT
            dataframe containing info on group means and weights for a given column

        column: str
            column to regularise

        weights_column: str
            name of weights column

        response_column: str
            name of response column

        Returns
        -------
        regularised : nw.Series
            Series of regularised encoding values
        """
        self.check_is_fitted(["global_mean"])

        prior_col = "prior_encodings"

        prior_df = group_means_and_weights.select(
            nw.col(column),
            (
                ((nw.col(response_column)) + self.global_mean * self.prior)
                / (nw.col(weights_column) + self.prior)
            ).alias(prior_col),
        )

        prior_encodings_dict = prior_df.to_dict(as_series=False)

        # return as dict
        return dict(
            zip(
                prior_encodings_dict[column],
                prior_encodings_dict[prior_col],
            ),
        )

    @nw.narwhalify
    def _fit_binary_response(
        self,
        X_y: FrameT,
        columns: list[str],
        weights_column: str,
        response_column: str,
    ) -> None:
        """Function to learn the MRE mappings for a given binary or continuous response.

        Parameters
        ----------
        X_y : pd/pl.DataFrame
            Data with catgeorical variable columns to transform, and target to encode against.

        columns : list(str)
            Post transform names of columns to be encoded. In the binary or continous case
            this is just self.columns. In the multi-level case this should be of the form
            {column_in_original_data}_{response_level}, where response_level is the level
            being encoded against in this call of _fit_binary_response.

        weights_column: str
            name of weights column

        response_column: str
            name of response column
        """
        # reuse mean imputer logic to calculate global mean
        mean_imputer = MeanImputer(
            columns=response_column,
            weights_column=weights_column,
        )
        mean_imputer.fit(X_y)
        self.global_mean = mean_imputer.impute_values_[response_column]

        X_y = X_y.with_columns(
            (nw.col(response_column) * nw.col(weights_column)).alias(response_column),
        )

        for c in columns:
            groupby_sum = X_y.group_by(c).agg(
                nw.col(response_column).sum(),
                nw.col(weights_column).sum(),
            )

            group_means_and_weights = groupby_sum.select(
                nw.col(response_column),
                nw.col(weights_column),
                nw.col(c),
            )

            self.mappings[c] = self._prior_regularisation(
                group_means_and_weights,
                column=c,
                weights_column=weights_column,
                response_column=response_column,
            )

            # to_dict changes types
            for key in self.mappings[c]:
                self.mappings[c][key] = self.cast_method(self.mappings[c][key])

    @nw.narwhalify
    def fit(self, X: FrameT, y: nw.Series) -> FrameT:
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
        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}
        self.unseen_levels_encoding_dict = {}

        if self.weights_column is not None:
            weights_column = self.weights_column
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)

        else:
            native_backend = nw.get_native_namespace(X)

            weights_column = "dummy_weights_column"
            X = X.with_columns(
                nw.new_series(
                    name=weights_column,
                    values=[1] * len(X),
                    backend=native_backend.__name__,
                ),
            )

        response_null_count = y.is_null().sum()

        if response_null_count > 0:
            msg = f"{self.classname()}: y has {response_null_count} null values"
            raise ValueError(msg)

        X_y = nw.from_native(self._combine_X_y(X, y))
        response_column = "_temporary_response"

        self.response_levels = self.level

        if self.level == "all":
            self.response_levels = y.unique()

        elif self.level is not None and any(
            level not in list(y.unique()) for level in self.level
        ):
            msg = "Levels contains a level to encode against that is not present in the response."
            raise ValueError(msg)

        self.column_to_encoded_columns = {column: [] for column in self.columns}
        self.encoded_columns = []

        levels_to_iterate_through = (
            self.response_levels
            if self.response_levels is not None
            # if no levels, just create arbitrary len 1 iterable
            else ["NO_LEVELS"]
        )

        for level in levels_to_iterate_through:
            # if multi level, set up placeholder encoded columns
            # if not multi level, will just overwrite existing column
            mapping_columns_for_this_level = {
                column: column + "_" + str(level) if self.MULTI_LEVEL else column
                for column in self.columns
            }

            X_y = X_y.with_columns(
                nw.col(column).alias(mapping_columns_for_this_level[column])
                for column in mapping_columns_for_this_level
            )

            # create temporary single level response columns to individually encode
            # if nans are present then will error in previous handling, so can assume not here
            X_y = X_y.with_columns(
                (y == level if self.MULTI_LEVEL else y).alias(response_column),
            )

            self._fit_binary_response(
                X_y,
                list(mapping_columns_for_this_level.values()),
                weights_column=weights_column,
                response_column=response_column,
            )

            self.column_to_encoded_columns = {
                column: [
                    *self.column_to_encoded_columns[column],
                    mapping_columns_for_this_level[column],
                ]
                for column in self.columns
            }

        self.encoded_columns = [
            value
            for column in self.columns
            for value in self.column_to_encoded_columns[column]
        ]

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
        self.null_mappings = base_mapping_transformer.null_mappings
        self.return_dtypes = base_mapping_transformer.return_dtypes

        self._fit_unseen_level_handling_dict(X_y, weights_column)

        return self

    @nw.narwhalify
    def _fit_unseen_level_handling_dict(
        self,
        X_y: FrameT,
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

        weights_column : str
            name of weights column

        """

        if isinstance(self.unseen_level_handling, (int, float)):
            for c in self.encoded_columns:
                self.unseen_levels_encoding_dict[c] = self.cast_method(
                    self.unseen_level_handling,
                )

        elif isinstance(self.unseen_level_handling, str):
            X_temp = nw.from_native(BaseMappingTransformMixin.transform(self, X_y))

            for c in self.encoded_columns:
                if self.unseen_level_handling in ["mean", "median"]:
                    group_weights = X_temp.group_by(c).agg(
                        nw.col(weights_column).sum(),
                    )

                    if self.unseen_level_handling == "mean":
                        # reuse MeanImputer logic to calculate means
                        mean_imputer = MeanImputer(
                            columns=c,
                            weights_column=weights_column,
                        )

                        mean_imputer.fit(group_weights)

                        self.unseen_levels_encoding_dict[
                            c
                        ] = mean_imputer.impute_values_[c]

                    # else, median
                    else:
                        # reuse MedianImputer logic to calculate medians
                        median_imputer = MedianImputer(
                            columns=c,
                            weights_column=weights_column,
                        )

                        median_imputer.fit(group_weights)

                        self.unseen_levels_encoding_dict[
                            c
                        ] = median_imputer.impute_values_[c]

                # else, min or max, which don't care about weights
                else:
                    self.unseen_levels_encoding_dict[c] = getattr(
                        X_temp[c],
                        self.unseen_level_handling,
                    )()

                self.unseen_levels_encoding_dict[c] = self.cast_method(
                    self.unseen_levels_encoding_dict[c],
                )

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
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

        # BaseTransformer.transform as we do not want to run check_mappable_rows in BaseNominalTransformer
        # (it causes complications with unseen levels and new cols, so run later)
        X = nw.from_native(BaseTransformer.transform(self, X))

        # create new columns, replacing unseen values with None
        # (they will be filled later)
        X = X.with_columns(
            nw.when(
                nw.col(old_column).is_in(self.mappings[new_column].keys()),
            )
            .then(
                nw.col(old_column),
            )
            .otherwise(
                None,
            )
            .alias(new_column)
            for old_column in self.columns
            for new_column in self.column_to_encoded_columns[old_column]
        )

        # with columns created, can now run parent transforms
        if self.unseen_level_handling:
            # do not want to run check_mappable_rows in this case, as will not like unseen values
            self.check_is_fitted(["unseen_levels_encoding_dict"])
        else:
            # temp overwrite columns attr so that check_mappable_rows runs safely
            original_columns = self.columns
            self.columns = self.encoded_columns
            X = nw.from_native(super().transform(X))
            self.columns = original_columns

        # next apply mappings
        X = nw.from_native(BaseMappingTransformMixin.transform(self, X))

        if self.unseen_level_handling:
            for old_col in self.columns:
                X = X.with_columns(
                    nw.when(
                        nw.col(new_col).is_in(list(self.mappings[new_col].values())),
                    )
                    .then(
                        nw.col(new_col),
                    )
                    .otherwise(self.unseen_levels_encoding_dict[new_col])
                    for new_col in self.column_to_encoded_columns[old_col]
                )

        columns_to_drop = [
            col for col in self.columns if col not in self.encoded_columns
        ]

        return DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            columns_to_drop,
        )


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
        self.null_mappings = base_mapping_transformer.null_mappings
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

    copy : bool, default = True
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

    def __init__(
        self,
        columns: str | list[str] | None = None,
        wanted_values: dict[str, list[str]] | None = None,
        separator: str = "_",
        drop_original: bool = False,
        copy: bool | None = None,
        verbose: bool = False,
        **kwargs: dict[str, bool],
    ) -> None:
        BaseTransformer.__init__(
            self,
            columns=columns,
            verbose=verbose,
            copy=copy,
            **kwargs,
        )

        if wanted_values is not None:
            if not isinstance(wanted_values, dict):
                msg = f"{self.classname()}: wanted_values should be a dictionary"
                raise TypeError(msg)

            for key, val_list in wanted_values.items():
                # check key is a string
                if not isinstance(key, str):
                    msg = f"{self.classname()}:  Key in 'wanted_values' should be a string"
                    raise TypeError(msg)

                # check value is a list
                if not isinstance(val_list, list):
                    msg = f"{self.classname()}: Values in the 'wanted_values' dictionary should be a list"
                    raise TypeError(msg)

                # check if each value within the list is a string
                for val in val_list:
                    if not isinstance(val, str):
                        msg = f"{self.classname()}: Entries in 'wanted_values' list should be a string"
                        raise TypeError(msg)

        self.wanted_values = wanted_values
        self.set_drop_original_column(drop_original)
        self.check_and_set_separator_column(separator)

    @nw.narwhalify
    def fit(self, X: FrameT, y: nw.Series | None = None) -> FrameT:
        """Gets list of levels for each column to be transformed. This defines which dummy columns
        will be created in transform.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to identify levels from.

        y : None
            Ignored. This parameter exists only for compatibility with sklearn.pipeline.Pipeline.

        """
        BaseTransformer.fit(self, X=X, y=y)

        # Check for nulls
        for c in self.columns:
            if X.select(nw.col(c).is_null().sum()).item() > 0:
                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c,
                )

        self.categories_ = {}
        self.new_feature_names_ = {}
        # Check each field has less than 100 categories/levels
        missing_levels = {}
        for c in self.columns:
            levels = X.select(nw.col(c).unique())

            level_count = levels.select(nw.col(c).count()).item()

            if level_count > 100:
                raise ValueError(
                    f"{self.classname()}: column %s has over 100 unique values - consider another type of encoding"
                    % c,
                )

            # Set the categories_ attribute to ensure check_is_fitted works
            levels_list = levels.get_column(c).to_list()

            # for consistency
            levels_list.sort()

            # categories if 'values' is provided
            selected_values = (
                self.wanted_values.get(c, None) if self.wanted_values else None
            )

            if selected_values is None:
                final_categories = levels_list
            else:
                final_categories = selected_values

            self.categories_[c] = final_categories
            self.new_feature_names_[c] = self._get_feature_names(column=c)

            present_levels = set(X.get_column(c).unique().to_list())
            missing_levels = self._warn_missing_levels(
                present_levels,
                c,
                missing_levels,
            )

        return self

    def _warn_missing_levels(
        self,
        present_levels: list,
        c: str,
        missing_levels: dict[str, list[str]],
    ) -> dict[str, list[str]]:
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
        missing_levels[c] = list(
            set(self.categories_[c]).difference(present_levels),
        )
        if len(missing_levels[c]) > 0:
            warning_msg = f"{self.classname()}: column {c} includes user-specified values {missing_levels[c]} not found in the dataset"
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

        return missing_levels

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

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Create new dummy columns from categorical fields.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to apply one hot encoding to.

        Returns
        -------
        X_transformed : pd/pl.DataFrame
            Transformed input X with dummy columns derived from categorical columns added. If drop_original
            = True then the original categorical columns that the dummies are created from will not be in
            the output X.

        """

        # Check that transformer has been fit before calling transform
        self.check_is_fitted(["categories_"])

        X = nw.from_native(BaseTransformer.transform(self, X))

        missing_levels = {}
        for c in self.columns:
            # Check for nulls
            if X.select(nw.col(c).is_null().sum()).item() > 0:
                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c,
                )

            # print warning for unseen levels
            present_levels = set(X.get_column(c).unique().to_list())
            unseen_levels = present_levels.difference(set(self.categories_[c]))
            if len(unseen_levels) > 0:
                warning_msg = f"{self.classname()}: column {c} has unseen categories: {unseen_levels}"
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

            # print warning for missing levels
            self._warn_missing_levels(present_levels, c, missing_levels)

            dummies = X.get_column(c).to_dummies(separator=self.separator)

            # insert 0 cols for missing levels
            dummies = dummies.with_columns(
                nw.lit(0).alias(c + self.separator + str(missing_level))
                for missing_level in missing_levels[c]
            )

            wanted_dummies = self.new_feature_names_[c]

            # cast dummy columns to bool
            dummies = dummies.with_columns(
                nw.col(new_column).cast(nw.Boolean) for new_column in wanted_dummies
            )
            X = nw.concat([X, dummies.select(wanted_dummies)], how="horizontal")

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
        )
