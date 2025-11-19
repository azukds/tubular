"""This module contains transformers that apply numeric functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, DataFrameMethodTransformer
from tubular.mixins import (
    CheckNumericMixin,
    DropOriginalMixin,
)
from tubular.types import (
    DataFrame,
    FloatTypeAnnotated,
    GenericKwargs,
    ListOfOneStr,
    ListOfTwoStrs,
    PositiveNumber,
)

if TYPE_CHECKING:
    from narwhals.typing import FrameT, IntoSeriesT


class BaseNumericTransformer(BaseTransformer, CheckNumericMixin):
    """
    Extends BaseTransformer for datetime scenarios.

    Parameters
    ----------
    columns : List[str]
        List of columns to be operated on.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    columns : List[str]
        List of columns to be operated on

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> BaseNumericTransformer(
    ... columns='a',
    ...    )
    BaseNumericTransformer(columns=['a'])

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = False

    FITS = False

    def __init__(self, columns: list[str], **kwargs: dict[str, bool]) -> None:
        super().__init__(columns=columns, **kwargs)

    @nw.narwhalify
    def fit(
        self,
        X: FrameT,
        y: nw.Series | None = None,
    ) -> BaseNumericTransformer:
        """Base fit method. Validates data and attributes prior to the child objects fit logic.

        Parameters
        ----------
        X : pd/pl.DataFrame
            A dataframe containing the required columns

        y : pd/pl.Series | None
            Required for pipeline.

        Example:
        --------
        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ... columns='a',
        ...    )

        >>> test_df = pl.DataFrame({'a': [1,2], 'b': [3,4]})

        >>> transformer.fit(test_df)
        BaseNumericTransformer(columns=['a'])

        """

        super().fit(X, y)

        CheckNumericMixin.check_numeric_columns(self, X[self.columns])

        return self

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Base transform method. Validates data and attributes prior to the child objects tranform logic.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Data to transform.

        return_native_override: Optional[bool]
            Option to override return_native attr in transformer, useful when calling parent
            methods
        Returns
        -------
        X : pd/pl.DataFrame
            Validated data

        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ... columns='a',
        ...    )

        >>> test_df = pl.DataFrame({'a': [1,2], 'b': [3,4]})

        >>> # base class has no effect on datag
        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        """
        X = _convert_dataframe_to_narwhals(X)
        return_native = self._process_return_native(return_native_override)
        X = super().transform(X, return_native_override=False)

        CheckNumericMixin.check_numeric_columns(self, X.select(self.columns))

        return _return_narwhals_or_native_dataframe(X, return_native)


class OneDKmeansTransformer(BaseNumericTransformer, DropOriginalMixin):
    """Transformer that generates a new column based on kmeans algorithm.
    Transformer runs the kmeans algorithm based on given number of clusters and then identifies the bins' cuts based on the results.
    Finally it passes them into the a cut function.

    Parameters
    ----------
    column : str
        Name of the column to discretise.

    new_column_name : str
        Name given to the new discrete column.

    n_clusters : int, default = 8
        The number of clusters to form as well as the number of centroids to generate.

    n_init "auto" or int, default="auto"
        Number of times the k-means algorithm is run with different centroid seeds.
        The final results is the best output of n_init consecutive runs in terms of inertia.
        Several runs are recommended for sparse high-dimensional problems (see `Clustering sparse data with k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#kmeans-sparse-high-dim>`__).

        When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable;
        1 if using init='k-means++' or init is an array-like.(Init is an arg in kmeans_kwargs. If init is not set then it defaults to k-means++ so n_init defaults to 1)

    drop_original : bool, default=False
        Should the original columns to be transformed be dropped after applying the
        OneDKmeanstransformer?

    kmeans_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the sklearn KMeans method when it is called in fit.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    Attributes
    ----------

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    --------
    >>> OneDKmeansTransformer(
    ... columns='a',
    ... n_clusters=2,
    ... new_column_name="new",
    ... drop_original=False,
    ... kmeans_kwargs={"random_state": 42},
    ...    )
    OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                          n_clusters=2, new_column_name='new')

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = False

    FITS = True

    @beartype
    def __init__(
        self,
        columns: Union[
            str,
            ListOfOneStr,
        ],
        new_column_name: str,
        n_init: Union[str, int] = "auto",
        n_clusters: int = 8,
        drop_original: bool = False,
        kmeans_kwargs: Optional[dict[str, object]] = None,
        **kwargs: bool,
    ) -> None:
        if kmeans_kwargs is None:
            kmeans_kwargs = {}

        self.n_clusters = n_clusters
        self.new_column_name = new_column_name
        self.n_init = n_init
        self.kmeans_kwargs = kmeans_kwargs
        self.drop_original = drop_original

        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        super().__init__(columns=[columns], **kwargs)

    def get_feature_names_out(self) -> list[str]:
        """list features modified/created by the transformer

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------

        >>> transformer = OneDKmeansTransformer(
        ... columns='a',
        ... n_clusters=2,
        ... new_column_name="kmeans_column",
        ... drop_original=False,
        ... kmeans_kwargs={"random_state": 42},
        ...  )

        >>> transformer.get_feature_names_out()
        ['kmeans_column']
        """

        return [
            self.new_column_name,
        ]

    @nw.narwhalify
    def fit(self, X: FrameT, y: IntoSeriesT | None = None) -> OneDKmeansTransformer:
        """Fit transformer to input data.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        Example:
        --------
        >>> import polars as pl

        >>> transformer=OneDKmeansTransformer(
        ... columns='a',
        ... n_clusters=2,
        ... new_column_name="new",
        ... drop_original=False,
        ... kmeans_kwargs={"random_state": 42},
        ...    )

        >>> test_df=pl.DataFrame({'a': [1,2,3,4],  'b': [5,6,7,8]})

        >>> transformer.fit(test_df)
        OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                              n_clusters=2, new_column_name='new')

        """

        super().fit(X, y)

        X = nw.from_native(X)

        # Check that X does not contain Nans and return ValueError.
        if (
            X.select(nw.col(self.columns[0]).is_null().any()).to_numpy().ravel()[0]
            or X.select(nw.col(self.columns[0]).is_nan().any()).to_numpy().ravel()[0]
        ):
            msg = f"{self.classname()}: X should not contain missing values."
            raise ValueError(msg)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            **self.kmeans_kwargs,
        )

        native_backend = nw.get_native_namespace(X).__name__
        groups = kmeans.fit_predict(X.select(self.columns[0]).to_numpy())

        X = X.with_columns(
            nw.new_series(
                name="groups",
                values=np.copy(groups),
                backend=native_backend,
            ),
        )

        self.bins = (
            X.group_by("groups")
            .agg(
                nw.col(self.columns[0]).max(),
            )
            .sort(self.columns[0])
            .select(self.columns[0])
            .to_numpy()
            .ravel()
        )
        return self

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Generate from input pd/pl.DataFrame (X) bins based on Kmeans results and add this column or columns in X.

        Parameters
        ----------
        X : pl/pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pl/pd.DataFrame
            Input X with additional cluster column added.

        Example:
        --------
        >>> import polars as pl

        >>> transformer=OneDKmeansTransformer(
        ... columns='a',
        ... n_clusters=2,
        ... new_column_name="new",
        ... drop_original=False,
        ... kmeans_kwargs={"random_state": 42},
        ...    )

        >>> test_df=pl.DataFrame({'a': [1,2,3,4],  'b': [5,6,7,8]})

        >>> _=transformer.fit(test_df)
        >>> transformer.transform(test_df)
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ new │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 5   ┆ 0   │
        │ 2   ┆ 6   ┆ 0   │
        │ 3   ┆ 7   ┆ 0   │
        │ 4   ┆ 8   ┆ 1   │
        └─────┴─────┴─────┘
        """
        X = super().transform(X)

        X = nw.from_native(X)
        native_backend = nw.get_native_namespace(X).__name__

        groups = np.digitize(
            X.select(self.columns[0]).to_numpy().ravel(),
            bins=self.bins,
            right=True,
        )

        X = X.with_columns(
            nw.new_series(
                name=self.new_column_name,
                values=groups,
                backend=native_backend,
            ),
        )
        return self.drop_original_column(X, self.drop_original, self.columns[0])


class DifferenceTransformer(BaseNumericTransformer):
    """Transformer that performs subtraction operation between two columns.

    This transformer allows performing subtraction between two columns in a DataFrame
    and stores the result in a new column.

    Attributes
    ----------
    columns : ListOfTwoStrs
        List of exactly two column names to operate on. The second column is subtracted from the first.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example
    -------
    >>> transformer = DifferenceTransformer(columns=['a', 'b'])
    >>> transformer.columns
    ['a', 'b']
    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = False

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialize the DifferenceTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The second column is subtracted from the first.
        verbose : bool, default=False
            Whether to print verbose output during transformation.
        """
        super().__init__(columns=columns, **kwargs)

        # Set new_column_name or generate a default one
        self.new_column_name = f"{columns[0]}_minus_{columns[1]}"

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the subtraction operation between two columns.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Transformed DataFrame with the new column containing the subtraction results.


        Example:
        --------
        >>> import polars as pl
        >>> transformer = DifferenceTransformer(columns=['a', 'b'])
        >>> test_df = pl.DataFrame({'a': [100, 200, 300], 'b': [80, 150, 200]})
        >>> transformer.transform(test_df)
        shape: (3, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ a_minus_b │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ i64 ┆ i64       │
        ╞═════╪═════╪═══════════╡
        │ 100 ┆ 80  ┆ 20        │
        │ 200 ┆ 150 ┆ 50        │
        │ 300 ┆ 200 ┆ 100       │
        └─────┴─────┴───────────┘
        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        # Create the subtraction expression
        expr = nw.col(self.columns[0]) - nw.col(self.columns[1])

        X = X.with_columns(expr.alias(self.new_column_name))

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.
        """
        return [f"{self.columns[0]}_minus_{self.columns[1]}"]


class RatioTransformer(BaseNumericTransformer):
    """Transformer that performs division operation between two columns.

    This transformer allows performing division between two columns in a DataFrame
    and stores the result in a new column.

    Attributes
    ----------
    columns : ListOfTwoStrs
        List of exactly two column names to operate on. The first column is the numerator,
        and the second column is the denominator.
    return_dtype : str
        The dtype of the resulting column, either 'Float32' or 'Float64'.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example
    -------
    >>> transformer = RatioTransformer(columns=['a', 'b'], return_dtype='Float32')
    >>> transformer.columns
    ['a', 'b']
    >>> transformer.return_dtype
    'Float32'
    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = False

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        >>> ratio_transformer = RatioTransformer(columns=['a', 'b'], return_dtype='Float32')
        >>> ratio_transformer.to_json()
        {'tubular_version': ..., 'classname': 'RatioTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'return_dtype': 'Float32'}, 'fit': {}}
        """

        json_dict = super().to_json()
        json_dict["init"]["return_dtype"] = self.return_dtype

        return json_dict

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        return_dtype: FloatTypeAnnotated = "Float32",
        **kwargs: Optional[bool],
    ) -> None:
        """Initialize the RatioTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The first column is the numerator,
            and the second column is the denominator.
        return_dtype : str, default='Float32'
            The dtype of the resulting column, either 'Float32' or 'Float64'.
        """
        super().__init__(columns=columns, **kwargs)

        self.return_dtype = return_dtype

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the division operation between two columns.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Transformed DataFrame with the new column containing the division results.

        Example:
        --------
        >>> import polars as pl
        >>> transformer = RatioTransformer(columns=['a', 'b'], return_dtype='Float32')
        >>> test_df = pl.DataFrame({'a': [100, 200, 300], 'b': [80, 150, 200]})
        >>> transformer.transform(test_df)
        shape: (3, 3)
        ┌─────┬─────┬────────────────┐
        │ a   ┆ b   ┆ a_divided_by_b │
        │ --- ┆ --- ┆ ---            │
        │ i64 ┆ i64 ┆ f32            │
        ╞═════╪═════╪════════════════╡
        │ 100 ┆ 80  ┆ 1.25           │
        │ 200 ┆ 150 ┆ 1.333333       │
        │ 300 ┆ 200 ┆ 1.5            │
        └─────┴─────┴────────────────┘
        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        # Create the division expression
        expr = (
            nw.when(nw.col(self.columns[1]) != 0)
            .then(nw.col(self.columns[0]) / nw.col(self.columns[1]))
            .otherwise(None)
            .cast(getattr(nw, self.return_dtype))
        )

        # Add the new column
        new_column_name = f"{self.columns[0]}_divided_by_{self.columns[1]}"
        X = X.with_columns(expr.alias(new_column_name))

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.
        """
        return [f"{self.columns[0]}_divided_by_{self.columns[1]}"]


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class LogTransformer(BaseNumericTransformer, DropOriginalMixin):
    """Transformer to apply log transformation.

    Transformer has the option to add 1 to the columns to log and drop the
    original columns.

    Parameters
    ----------
    columns : None or str or list
        Columns to log transform.

    base : None or float/int
        Base for log transform. If None uses natural log.

    add_1 : bool
        Should a constant of 1 be added to the columns to be transformed prior to
        applying the log transform?

    drop_original : bool
        Should the original columns to be transformed be dropped after applying the
        log transform?

    suffix : str, default = '_log'
        The suffix to add onto the end of column names for new columns.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    add_1 : bool
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    drop_original : bool
        The name of the pandas.DataFrame method to call.

    suffix : str
        The suffix to add onto the end of column names for new columns.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    FITS = False

    @beartype
    def __init__(
        self,
        columns: Optional[Union[str, list[str]]],
        base: Optional[PositiveNumber] = None,
        add_1: bool = False,
        drop_original: bool = True,
        suffix: str = "log",
        **kwargs: bool,
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        self.drop_original = drop_original
        self.base = base
        self.add_1 = add_1
        self.suffix = suffix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log transform to the specified columns.

        If the drop attribute is True then the original columns are dropped. If
        the add_1 attribute is True then the original columns + 1 are logged.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The dataframe with the specified columns logged, optionally dropping the original
            columns if self.drop is True.

        """
        X = super().transform(X)

        new_column_names = [f"{column}_{self.suffix}" for column in self.columns]

        if self.add_1:
            if (X[self.columns] <= -1).sum().sum() > 0:
                msg = f"{self.classname()}: values less than or equal to 0 in columns (after adding 1), make greater than 0 before using transform"
                raise ValueError(msg)

            if self.base is None:
                X[new_column_names] = np.log1p(X[self.columns])

            else:
                X[new_column_names] = np.log1p(X[self.columns]) / np.log(self.base)

        else:
            if (X[self.columns] <= 0).sum().sum() > 0:
                msg = f"{self.classname()}: values less than or equal to 0 in columns, make greater than 0 before using transform"
                raise ValueError(msg)

            if self.base is None:
                X[new_column_names] = np.log(X[self.columns])

            else:
                X[new_column_names] = np.log(X[self.columns]) / np.log(self.base)

        return self.drop_original_column(X, self.drop_original, self.columns)


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class CutTransformer(BaseNumericTransformer):
    """Class to bin a column into discrete intervals.

    Class simply uses the [pd.cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)
    method on the specified column.

    Parameters
    ----------
    column : str
        Name of the column to discretise.

    new_column_name : str
        Name given to the new discrete column.

    cut_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.cut method when it is called in transform.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    Attributes
    ----------

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    FITS = False

    @beartype
    def __init__(
        self,
        column: str,
        new_column_name: str,
        cut_kwargs: Optional[GenericKwargs] = None,
        **kwargs: bool,
    ) -> None:
        if cut_kwargs is None:
            cut_kwargs = {}

        self.cut_kwargs = cut_kwargs
        self.new_column_name = new_column_name

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

        super().__init__(columns=[column], **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Discretise specified column using pd.cut.

        Parameters
        ----------
        X : pd.DataFrame
            Data with column to transform.

        """
        X = super().transform(X)

        # quick fix for empty frames, not spending much
        # time on this as transformer is deprecated
        if X.empty:
            X[self.new_column_name] = pd.Series(dtype=float)

        else:
            X[self.new_column_name] = pd.cut(
                X[self.columns[0]].to_numpy(),
                **self.cut_kwargs,
            )

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If aspects of it have been useful to you, please raise an issue
    for it to be replaced with more specific transformers
    """,
)
class TwoColumnOperatorTransformer(
    DataFrameMethodTransformer,
    BaseNumericTransformer,
):
    """This transformer applies a pandas.DataFrame method to two columns (add, sub, mul, div, mod, pow).

    Transformer assigns the output of the method to a new column. The method will be applied
    in the form (column 1)operator(column 2), so order matters (if the method does not commute). It is possible to
    supply other key word arguments to the transform method, which will be passed to the pandas.DataFrame method being called.

    Parameters
    ----------
    pd_method_name : str
        The name of the pandas.DataFrame method to be called.

    column1_name : str
        The name of the 1st column in the operation.

    column2_name : str
        The name of the 2nd column in the operation.

    new_column_name : str
        The name of the new column that the output is assigned to.

    pd_method_kwargs : dict, default =  {'axis':0}
        Dictionary of method kwargs to be passed to pandas.DataFrame method. Must contain an entry for axis, set to either 1 or 0.

    **kwargs :
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    pd_method_name : str
        The name of the pandas.DataFrame method to be called.

    columns : list
        list containing two string items: [column1_name, column2_name] The first will be operated upon by the
        chosen pandas method using the second.

    column2_name : str
        The name of the 2nd column in the operation.

    new_column_name : str
        The name of the new column that the output is assigned to.

    pd_method_kwargs : dict
        Dictionary of method kwargs to be passed to pandas.DataFrame method.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    FITS = False

    @beartype
    def __init__(
        self,
        pd_method_name: str,
        columns: ListOfTwoStrs,
        new_column_name: str,
        pd_method_kwargs: Optional[dict[str, object]] = None,
        **kwargs: Optional[bool],
    ) -> None:
        """Performs input checks not done in either DataFrameMethodTransformer.__init__ or BaseTransformer.__init__."""
        if pd_method_kwargs is None:
            pd_method_kwargs = {"axis": 0}
        else:
            if "axis" not in pd_method_kwargs:
                msg = f'{self.classname()}: pd_method_kwargs must contain an entry "axis" set to 0 or 1'
                raise ValueError(msg)
            if pd_method_kwargs["axis"] not in [0, 1]:
                msg = f"{self.classname()}: pd_method_kwargs 'axis' must be 0 or 1"
                raise ValueError(msg)

        self.new_column_name = new_column_name

        # call DataFrameMethodTransformer.__init__
        # This class will inherit all the below attributes from DataFrameMethodTransformer
        super().__init__(
            new_column_names=new_column_name,
            pd_method_name=pd_method_name,
            columns=columns,
            pd_method_kwargs=pd_method_kwargs,
            **kwargs,
        )

        self.column1_name = columns[0]
        self.column2_name = columns[1]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data by applying the chosen method to the two specified columns.

        Args:
        ----
            X (pd.DataFrame): Data to transform.

        Returns:
        -------
            pd.DataFrame: Input X with an additional column.
        """
        # call appropriate parent transforms
        X = super(DataFrameMethodTransformer, self).transform(X)
        X = super(BaseNumericTransformer, self).transform(X)

        X[self.new_column_name] = getattr(X[[self.column1_name]], self.pd_method_name)(
            X[self.column2_name],
            **self.pd_method_kwargs,
        )

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class ScalingTransformer(BaseNumericTransformer):
    """Transformer to perform scaling of numeric columns.

    Transformer can apply min max scaling, max absolute scaling or standardisation (subtract mean and divide by std).
    The transformer uses the appropriate sklearn.preprocessing scaler.

    Parameters
    ----------
    columns : str, list or None
        Name of the columns to apply scaling to.

    scaler_type : str
        Type of scaler to use, must be one of 'min_max', 'max_abs' or 'standard'. The corresponding
        sklearn.preprocessing scaler used in each case is MinMaxScaler, MaxAbsScaler or StandardScaler.

    scaler_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the scaler object when it is initialised.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    Attributes
    ----------

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    FITS = True

    # Dictionary mapping scaler types to their corresponding sklearn classes
    scaler_options: ClassVar[
        dict[str, Union[MinMaxScaler, MaxAbsScaler, StandardScaler]]
    ] = {
        "min_max": MinMaxScaler,
        "max_abs": MaxAbsScaler,
        "standard": StandardScaler,
    }

    def __init__(
        self,
        columns: str | list[str] | None,
        scaler_type: str,
        scaler_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if scaler_kwargs is None:
            scaler_kwargs = {}

        # Validate scaler_kwargs type
        if not isinstance(scaler_kwargs, dict):
            msg = f"{self.classname()}: scaler_kwargs should be a dict but got type {type(scaler_kwargs)}"
            raise TypeError(msg)

        for i, k in enumerate(scaler_kwargs.keys()):
            if not isinstance(k, str):
                msg = f"{self.classname()}: unexpected type ({type(k)}) for scaler_kwargs key in position {i}, must be str"
                raise TypeError(msg)

        # Validate scaler_type
        if scaler_type not in self.scaler_options:
            allowed_scaler_values = list(self.scaler_options.keys())
            msg = f"{self.classname()}: scaler_type should be one of; {allowed_scaler_values}"
            raise ValueError(msg)

        # Initialize scaler using the dictionary
        self.scaler = self.scaler_options[scaler_type](**scaler_kwargs)
        # This attribute is not for use in any method
        # Here only as a fix to allow string representation of transformer.
        self.scaler_kwargs = scaler_kwargs
        self.scaler_type = scaler_type

        super().__init__(columns=columns, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ScalingTransformer:
        """Fit scaler to input data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        """
        super().fit(X, y)
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data X with fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe containing columns to be scaled.

        Returns
        -------
        X : pd.DataFrame
            Input X with columns scaled.

        """
        X = super().transform(X)

        # quick fix for empty frames, not spending much
        # time on this as transformer is deprecated
        if X.empty:
            for col in self.columns:
                X[col] = pd.Series(dtype=float)

        else:
            X[self.columns] = self.scaler.transform(X[self.columns])

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class InteractionTransformer(BaseNumericTransformer):
    """Transformer that generates interaction features.
    Transformer generates a new column  for all combinations from the selected columns up to the maximum degree
    provided. (For sklearn version higher than 1.0.0>, only interaction of a degree higher or equal to the minimum
    degree would be computed).
    Each interaction column consists of the product of the specific combination of columns.
    Ex: with 3 columns provided ["a","b","c"], if max degree is 3, the total possible combinations are :
    - of degree 1 : ["a","b","c"]
    - of degree 2 : ["a b","b c","a c"]
    - of degree 3 : ["a b c"].

    Parameters
    ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. This is avoid likely
            when the user forget to set columns, in this case all columns would be picked up when super
            transform runs.
        min_degree : int
            minimum degree of interaction features to be considered. For example if min_degree=3, only interaction
            columns from at least 3 columns would be generated. NB- only applies if sklearn version is 1.0.0>=
        max_degree : int
            maximum degree of interaction features to be considered. For example if max_degree=3, only interaction
            columns from up to 3 columns would be generated.


    Attributes
    ----------
        min_degree : int
            minimum degree of interaction features to be considered

        max_degree : int
            maximum degree of interaction features to be considered

        nb_features_to_interact : int
            number of selected columns from which interactions should be computed. (=len(columns))

        nb_combinations : int
            number of new interaction features

        interaction_colname : list
            names of each new interaction feature. The name of an interaction feature is the combinations of previous
            column names joined with a whitespace. Interaction feature of ["col1","col2","col3] would be "col1 col2 col3".

        nb_feature_out : int
            number of total columns of transformed dataset, including new interaction features

        built_from_json: bool
            indicates if transformer was reconstructed from json, which limits it's supported
            functionality to .transform

        polars_compatible : bool
            class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

        FITS: bool
            class attribute, indicates whether transform requires fit to be run first

        jsonable: bool
            class attribute, indicates if transformer supports to/from_json methods

        lazyframe_compatible: bool
            class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    FITS = False

    def __init__(
        self,
        columns: str | list[str] | None,
        min_degree: int = 2,
        max_degree: int = 2,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if len(columns) < 2:
            msg = f"{self.classname()}: number of columns must be equal or greater than 2, got {len(columns)} column."
            raise ValueError(msg)

        if type(min_degree) is int:
            if min_degree < 2:
                msg = f"{self.classname()}: min_degree must be equal or greater than 2, got {min_degree}"
                raise ValueError(msg)
            self.min_degree = min_degree
        else:
            msg = f"{self.classname()}: unexpected type ({type(min_degree)}) for min_degree, must be int"
            raise TypeError(msg)
        if type(max_degree) is int:
            if min_degree > max_degree:
                msg = f"{self.classname()}: max_degree must be equal or greater than min_degree"
                raise ValueError(msg)
            self.max_degree = max_degree
            if max_degree > len(columns):
                msg = f"{self.classname()}: max_degree must be equal or lower than number of columns"
                raise ValueError(msg)
            self.max_degree = max_degree
            if max_degree > len(columns):
                msg = f"{self.classname()}: max_degree must be equal or lower than number of columns"
                raise ValueError(msg)
            self.max_degree = max_degree
        else:
            msg = f"{self.classname()}: unexpected type ({type(max_degree)}) for max_degree, must be int"
            raise TypeError(msg)

        self.nb_features_to_interact = len(self.columns)
        self.nb_combinations = -1
        self.interaction_colname = []
        self.nb_feature_out = -1

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate from input pandas DataFrame (X) new interaction features using the "product" pandas.DataFrame method
         and add this column or columns in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.interaction_colname) added. These contain the output of
            running the  product pandas DataFrame method on identified combinations.

        """
        X = super().transform(X)

        try:
            interaction_combination_index = PolynomialFeatures._combinations(
                n_features=self.nb_features_to_interact,
                min_degree=self.min_degree,
                max_degree=self.max_degree,
                interaction_only=True,
                include_bias=False,
            )
        except TypeError as err:
            if (
                str(err)
                == "_combinations() got an unexpected keyword argument 'min_degree'"
            ):
                interaction_combination_index = PolynomialFeatures._combinations(
                    n_features=self.nb_features_to_interact,
                    degree=self.max_degree,
                    interaction_only=True,
                    include_bias=False,
                )
            else:
                raise err

        interaction_combination_colname = [
            [self.columns[col_idx] for col_idx in interaction_combination]
            for interaction_combination in interaction_combination_index
        ]
        self.nb_combinations = len(interaction_combination_colname)
        self.nb_feature_out = self.nb_combinations + len(X)

        self.interaction_colname = [
            " ".join(interaction_combination)
            for interaction_combination in interaction_combination_colname
        ]

        for inter_idx in range(len(interaction_combination_colname)):
            X[self.interaction_colname[inter_idx]] = X[
                interaction_combination_colname[inter_idx]
            ].product(axis=1, skipna=False)

        return X


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class PCATransformer(BaseNumericTransformer):
    """Transformer that generates variables using Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    It is based on sklearn class sklearn.decomposition.PCA

    Parameters
    ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. When the user forget to set columns,
            all columns would be picked up when super transform runs.
        n_components : int, float or 'mle', default=None
            Number of components to keep.
            if n_components is not set all components are kept::
                n_components == min(n_samples, n_features)
            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.
            If ``svd_solver == 'arpack'``, the number of components must be
             strictly less than the minimum of n_features and n_samples.
            Hence, the None case results in::
                n_components == min(n_samples, n_features) - 1   svd_solver='auto', tol=0.0,  n_oversamples=10, random_state=None
        svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            If auto :
                The solver is selected by a default policy based on `X.shape` and
                `n_components`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient 'randomized'
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full :
                run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing
            If arpack :
                run SVD truncated to n_components calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                0 < n_components < min(X.shape)
            If randomized :
                run randomized SVD by the method of Halko et al.
            .. sklearn versionadded:: 0.18.0

        random_state : int, RandomState instance or None, default=None
            Used when the 'arpack' or 'randomized' solvers are used. Pass an int
            for reproducible results across multiple function calls.
            .. sklearn versionadded:: 0.18.0
        pca_column_prefix : str, prefix added to each the n components features generated. Default is "pca_"
            example: if n_components = 3, new columns would be 'pca_0','pca_1','pca_2'.

    Attributes
    ----------
    pca : PCA class from sklearn.decomposition

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    feature_names_out: list or None
        list of feature name representing the new dimensions.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = False

    lazyframe_compatible = False

    jsonable = False

    FITS = True

    def __init__(
        self,
        columns: str | list[str] | None,
        n_components: int = 2,
        svd_solver: str = "auto",
        random_state: int | np.random.RandomState = None,
        pca_column_prefix: str = "pca_",
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if type(n_components) is int:
            if n_components < 1:
                msg = f"{self.classname()}:n_components must be strictly positive got {n_components}"
                raise ValueError(msg)
            self.n_components = n_components
        elif type(n_components) is float:
            if 0 < n_components < 1:
                self.n_components = n_components
            else:
                msg = f"{self.classname()}:n_components must be strictly positive and must be of type int when greater than or equal to 1. Got {n_components}"
                raise ValueError(msg)

        else:
            if n_components == "mle":
                self.n_components = n_components
            else:
                msg = f"{self.classname()}:unexpected type {type(n_components)} for n_components, must be int, float (0-1) or equal to 'mle'."
                raise TypeError(msg)

        if type(svd_solver) is str:
            if svd_solver not in ["auto", "full", "arpack", "randomized"]:
                msg = f"{self.classname()}:svd_solver {svd_solver} is unknown. Please select among 'auto', 'full', 'arpack', 'randomized'."
                raise ValueError(msg)
            self.svd_solver = svd_solver
        else:
            msg = f"{self.classname()}:unexpected type {type(svd_solver)} for svd_solver, must be str"
            raise TypeError(msg)

        if type(random_state) is int or random_state is None:
            self.random_state = random_state
        else:
            msg = f"{self.classname()}:unexpected type {type(random_state)} for random_state, must be int or None."
            raise TypeError(msg)

        if (svd_solver == "arpack") and (n_components == "mle"):
            msg = f"{self.classname()}: n_components='mle' cannot be a string with svd_solver='arpack'"
            raise ValueError(msg)
        if (svd_solver in ["randomized", "arpack"]) and (type(n_components) is float):
            msg = f"{self.classname()}: n_components {n_components} cannot be a float with svd_solver='{svd_solver}'"
            raise TypeError(msg)

        if type(pca_column_prefix) is str:
            self.pca_column_prefix = pca_column_prefix
        else:
            msg = f"{self.classname()}:unexpected type {type(pca_column_prefix)} for pca_column_prefix, must be str"
            raise TypeError(msg)

        self.pca = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

        self.pca_column_prefix = pca_column_prefix
        self.feature_names_out = None
        self.n_components_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit PCA to input data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        """
        super().fit(X, y)

        X = CheckNumericMixin.check_numeric_columns(self, X)

        if self.n_components != "mle":
            if 0 < self.n_components <= min(X[self.columns].shape):
                pass
            else:
                msg = f"{self.classname()}: n_components {self.n_components} must be between 1 and min(n_samples {X[self.columns].shape[0]}, n_features {X[self.columns].shape[1]}) is {min(X[self.columns].shape)} with svd_solver '{self.svd_solver}'"
                raise ValueError(msg)

        self.pca.fit(X[self.columns])
        self.n_components_ = self.pca.n_components_
        self.feature_names_out = [
            self.pca_column_prefix + str(i) for i in range(self.n_components_)
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate from input pandas DataFrame (X) PCA features and add this column or columns in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.interaction_colname) added. These contain the output of
            running the  product pandas DataFrame method on identified combinations.
        """
        X = super().transform(X)
        X = CheckNumericMixin.check_numeric_columns(self, X)

        # quick fix for empty frames, not spending much
        # time on this as transformer is deprecated
        if X.empty:
            for col in self.feature_names_out:
                X[col] = pd.Series(dtype=float)

        else:
            X[self.feature_names_out] = self.pca.transform(X[self.columns])

        return X
