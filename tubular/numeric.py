"""Contains transformers that apply numeric functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
from beartype import beartype
from sklearn.cluster import KMeans

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.numeric import (
    get_difference_of_two_columns,
    get_ratio_of_two_columns,
)
from tubular.mixins import (
    CheckNumericMixin,
)
from tubular.types import (
    DataFrame,
    FloatTypeAnnotated,
    ListOfOneStr,
    ListOfTwoStrs,
)

if TYPE_CHECKING:
    from narwhals.typing import FrameT, IntoSeriesT


@register
class BaseNumericTransformer(BaseTransformer, CheckNumericMixin):
    """Extends BaseTransformer for datetime scenarios.

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

    Examples
    --------
    ```pycon
    >>> BaseNumericTransformer(
    ...     columns="a",
    ... )
    BaseNumericTransformer(columns=['a'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = False

    FITS = False

    def __init__(self, columns: list[str], **kwargs: dict[str, bool]) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : List[str]
            List of columns to be operated on.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

    def fit(
        self,
        X: DataFrame,
        y: nw.Series | None = None,
    ) -> BaseNumericTransformer:
        """Validate data and attributes prior to the child objects fit logic.

        Parameters
        ----------
        X : DataFrame
            A dataframe containing the required columns

        y : Series | None
            Required for pipeline.

        Returns
        -------
            BaseNumericTransformer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ...     columns="a",
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        >>> transformer.fit(test_df)
        BaseNumericTransformer(columns=['a'])

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        super().fit(X, y)

        CheckNumericMixin.check_numeric_columns(self, X.select(self.columns))

        return self

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Validate data and attributes prior to the child objects transform logic.

        Parameters
        ----------
        X : DataFrame
            Data to transform.

        return_native_override: Optional[bool]
            Option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Validated data

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ...     columns="a",
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

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

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        return_native = self._process_return_native(return_native_override)
        X = super().transform(X, return_native_override=False)

        CheckNumericMixin.check_numeric_columns(self, X.select(self.columns))

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class OneDKmeansTransformer(BaseNumericTransformer):
    """Generates a new column based on kmeans algorithm.

    Transformer runs the kmeans algorithm based on given number of clusters and then identifies the bins' cuts based on the results.
    Finally it passes them into the a cut function.

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

    extractable_exprs: bool
        class attribute, indicates whether expressions for transformation can be
        extracted form fitted transformer using `get_transform_exprs` method

    Examples
    --------
    ```pycon
    >>> OneDKmeansTransformer(
    ...     columns="a",
    ...     n_clusters=2,
    ...     new_column_name="new",
    ...     kmeans_kwargs={"random_state": 42},
    ... )
    OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                          n_clusters=2, new_column_name='new')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    extractable_exprs = False

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        >>> import polars as pl
        >>> x = OneDKmeansTransformer(
        ... columns='a',
        ... n_clusters=2,
        ... new_column_name="new",
        ... kmeans_kwargs={"random_state": 42},
        ...    )
        >>> test_df=pl.DataFrame({'a': [1,2,3,4],  'b': [5,6,7,8]})
        >>> x.fit(test_df)
        OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                              n_clusters=2, new_column_name='new')
        >>> x.to_json()
        {'tubular_version': ..., 'classname': 'OneDKmeansTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'new', 'n_init': 'auto', 'n_clusters': 2, 'kmeans_kwargs': {'random_state': 42}}, 'fit': {'is_fitted_': True, 'bins': [3, 4]}}

        """
        self.check_is_fitted(["bins"])
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "new_column_name": self.new_column_name,
                "n_init": self.n_init,
                "n_clusters": self.n_clusters,
                "kmeans_kwargs": self.kmeans_kwargs,
            },
        )
        json_dict["fit"]["bins"] = self.bins

        return json_dict

    @beartype
    def __init__(
        self,
        columns: str | ListOfOneStr,
        new_column_name: str,
        n_init: str | int = "auto",
        n_clusters: int = 8,
        kmeans_kwargs: dict[str, object] | None = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list[str]
            Name of the column to discretise.

        new_column_name : str
            Name given to the new discrete column.

        n_clusters : int, default = 8
            The number of clusters to form as well as the number of centroids to generate.

        n_init: "auto" or int, default="auto"
            Number of times the k-means algorithm is run with different centroid seeds.
            The final results is the best output of n_init consecutive runs in terms of inertia.
            Several runs are recommended for sparse high-dimensional problems (see `Clustering sparse data with k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#kmeans-sparse-high-dim>`__).

            When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable;
            1 if using init='k-means++' or init is an array-like.(Init is an arg in kmeans_kwargs. If init is not set then it defaults to k-means++ so n_init defaults to 1)

        kmeans_kwargs : dict, default = {}
            A dictionary of keyword arguments to be passed to the sklearn KMeans method when it is called in fit.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init().

        """
        if kmeans_kwargs is None:
            kmeans_kwargs = {}

        self.n_clusters = n_clusters
        self.new_column_name = new_column_name
        self.n_init = n_init
        self.kmeans_kwargs = kmeans_kwargs

        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        super().__init__(columns=self.columns, **kwargs)

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="kmeans_column",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> transformer.get_feature_names_out()
        ['kmeans_column']

        ```

        """
        return [
            self.new_column_name,
        ]

    @block_from_json
    @nw.narwhalify
    def fit(self, X: FrameT, y: IntoSeriesT | None = None) -> OneDKmeansTransformer:
        """Fit transformer to input data.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        Returns
        -------
            OneDKmeansTransformer:
                Fitted class instance.

        Raises
        ------
        ValueError:
            if columns in X contain missing values.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="new",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

        >>> transformer.fit(test_df)
        OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                              n_clusters=2, new_column_name='new')

        ```

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
            .tolist()
        )
        self.is_fitted_ = True
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

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="new",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

        >>> _ = transformer.fit(test_df)
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

        ```

        """
        X = super().transform(X)

        X = nw.from_native(X)
        native_backend = nw.get_native_namespace(X).__name__

        groups = np.digitize(
            X.select(self.columns[0]).to_numpy().ravel(),
            bins=self.bins,
            right=True,
        )

        return X.with_columns(
            nw.new_series(
                name=self.new_column_name,
                values=groups,
                backend=native_backend,
            ),
        )


@register
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

    extractable_exprs: bool
        class attribute, indicates whether expressions for transformation can be
        extracted form fitted transformer using `get_transform_exprs` method

    Examples
    --------
    ```pycon
    >>> transformer = DifferenceTransformer(columns=["a", "b"])
    >>> transformer.columns
    ['a', 'b']

    ```

    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = True

    extractable_exprs = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        **kwargs: bool | None,
    ) -> None:
        """Initialize the DifferenceTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The second column is subtracted from the first.
        verbose : bool, default=False
            Whether to print verbose output during transformation.
        kwargs: bool
            arguments for base class, e.g. verbose.

        """
        super().__init__(columns=columns, **kwargs)

        # Set new_column_name or generate a default one
        self.new_column_name = f"{columns[0]}_minus_{columns[1]}"
        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return get_difference_of_two_columns(columns=self.columns)

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the subtraction operation between two columns.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        DataFrame
            Transformed DataFrame with the new column containing the subtraction results.


        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = DifferenceTransformer(columns=["a", "b"])
        >>> test_df = pl.DataFrame({"a": [100, 200, 300], "b": [80, 150, 200]})
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

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        transform_expr = self.get_transform_exprs()

        X = X.with_columns(transform_expr)

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.

        """
        return [f"{self.columns[0]}_minus_{self.columns[1]}"]


@register
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

    extractable_exprs: bool
        class attribute, indicates whether expressions for transformation can be
        extracted form fitted transformer using `get_transform_exprs` method

    Examples
    --------
    ```pycon
    >>> transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
    >>> transformer.columns
    ['a', 'b']
    >>> transformer.return_dtype
    'Float32'

    ```

    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = True

    extractable_exprs = True

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        ```pycon
        >>> ratio_transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
        >>> ratio_transformer.to_json()
        {'tubular_version': ..., 'classname': 'RatioTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'return_dtype': 'Float32'}, 'fit': {'is_fitted_': True}}

        ```

        """
        json_dict = super().to_json()
        json_dict["init"]["return_dtype"] = self.return_dtype

        return json_dict

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        return_dtype: FloatTypeAnnotated = "Float32",
        **kwargs: bool | None,
    ) -> None:
        """Initialize the RatioTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The first column is the numerator,
            and the second column is the denominator.
        return_dtype : str, default='Float32'
            The dtype of the resulting column, either 'Float32' or 'Float64'.
        kwargs: bool
            arguments for base class, e.g. verbose

        """
        super().__init__(columns=columns, **kwargs)

        self.return_dtype = return_dtype
        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return get_ratio_of_two_columns(
            columns=self.columns,
            return_dtype=self.return_dtype,
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the division operation between two columns.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        DataFrame
            Transformed DataFrame with the new column containing the division results.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
        >>> test_df = pl.DataFrame({"a": [100, 200, 300], "b": [80, 150, 200]})
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

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        transform_expr = self.get_transform_exprs()

        X = X.with_columns(transform_expr)

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.

        """
        return [f"{self.columns[0]}_divided_by_{self.columns[1]}"]
