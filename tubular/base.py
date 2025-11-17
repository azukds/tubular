"""Base transformer classes and utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import narwhals as nw
import pandas as pd
from beartype import beartype
from beartype.typing import Annotated, List
from beartype.vale import Is
from sklearn.base import BaseEstimator, TransformerMixin

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.mixins import DropOriginalMixin
from tubular.types import DataFrame

if TYPE_CHECKING:
    from narwhals.typing import FrameT

# Set pandas copy-on-write mode to avoid warnings
# This is a global setting that affects all pandas operations
# It's set here to ensure it's applied before any pandas operations
pd.options.mode.copy_on_write = True


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base tranformer class which all other transformers in the package inherit from.

    This class provides common functionality for all transformers including:
    - Column validation
    - DataFrame conversion (pandas/polars to narwhals)
    - JSON serialization support
    - Standard sklearn transformer interface

    Parameters
    ----------
    columns : str or list of str, optional
        Columns to transform. If None, all columns are used.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    columns : list of str
        Columns to transform.
    copy : bool
        Whether to copy the dataframe before transformation.
    verbose : bool
        Whether to print verbose output.
    return_native : bool
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Notes
    -----
    This is an abstract base class. Subclasses should implement the `transform`
    method to define their specific transformation logic.

    Examples
    --------
    >>> from tubular.base import BaseTransformer
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> transformer = BaseTransformer(columns='a')
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       a  b
    0  1  4
    1  2  5
    2  3  6

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = False

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    # class attribute, indicates whether transformer can be serialized to JSON
    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize BaseTransformer.

        Parameters
        ----------
        columns : str or list of str, optional
            Columns to transform. If None, all columns are used.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        self.columns = columns if columns is None else ([columns] if isinstance(columns, str) else columns)
        self.copy = copy
        self.verbose = verbose
        self.return_native = return_native

    def _check_columns_exist(self, X: DataFrame) -> None:
        """Check that specified columns exist in the dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe to check.

        Raises
        ------
        ValueError
            If any specified columns don't exist in the dataframe.

        """
        if self.columns is None:
            return

        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in dataframe")

    def _check_is_fitted(self) -> None:
        """Check if transformer has been fitted.

        Raises
        ------
        NotImplementedError
            If transformer requires fitting but hasn't been fitted yet.

        """
        if self.FITS and not hasattr(self, "_is_fitted"):
            raise NotImplementedError(
                f"{self.__class__.__name__} requires fitting before transform. "
                "Call fit() first."
            )

    @beartype
    def transform(
        self,
        X: Annotated[
            nw.TypedFrame,
            Is[lambda frame: nw.is_dataframe(frame) and nw.len(frame) > 0],
        ],
        return_native_override: Optional[bool] = None,
    ) -> nw.TypedFrame:
        """Transform the dataframe.

        This is the main transformation method that should be overridden by
        subclasses to implement their specific transformation logic.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.
        return_native_override : bool, optional
            Override the return_native setting for this transform call.

        Returns
        -------
        narwhals Frame
            Transformed dataframe.

        """
        self._check_is_fitted()
        self._check_columns_exist(X)

        X = _convert_dataframe_to_narwhals(X)

        if self.copy:
            X = X.clone()

        return_native = return_native_override if return_native_override is not None else self.return_native
        return _return_narwhals_or_native_dataframe(X, return_native)

    @block_from_json
    def to_json(self) -> dict[str, Any]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, Any]
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        >>> from tubular.base import BaseTransformer
        >>> transformer = BaseTransformer(columns='a', copy=True)
        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseTransformer', 'init': {'columns': ['a'], 'copy': True, 'verbose': False, 'return_native': True}, 'fit': {}}

        """
        import tubular

        return {
            "tubular_version": tubular.__version__,
            "classname": self.__class__.__name__,
            "init": {
                "columns": self.columns,
                "copy": self.copy,
                "verbose": self.verbose,
                "return_native": self.return_native,
            },
            "fit": {},
        }

    def columns_check(self, X: DataFrame) -> None:
        """Check that specified columns exist in the dataframe.

        This is a deprecated method. Use `_check_columns_exist` instead.

        Parameters
        ----------
        X : DataFrame
            Input dataframe to check.

        Raises
        ------
        ValueError
            If any specified columns don't exist in the dataframe.

        .. deprecated:: 0.5.0
            Use `_check_columns_exist` instead. This method will be removed
            in a future version.

        """
        warnings.warn(
            "columns_check is deprecated. Use _check_columns_exist instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._check_columns_exist(X)


@deprecated(
    "DataFrameMethodTransformer is deprecated and will be removed in a future version. "
    "Please use the specific transformer for your use case or create a custom transformer. "
    "If you have a specific use case, please open an issue for it to be redeveloped!",
)
class DataFrameMethodTransformer(DropOriginalMixin, BaseTransformer):
    """Tranformer that applies a pandas.DataFrame method.

    This transformer applies a specified pandas DataFrame method to the
    input dataframe. It's a generic transformer that can be used for any
    pandas method that operates on DataFrames.

    Parameters
    ----------
    method : str
        Name of the pandas DataFrame method to apply.
    method_args : dict, optional
        Arguments to pass to the method.
    columns : str or list of str, optional
        Columns to transform. If None, all columns are used.
    drop_original : bool, default=False
        Whether to drop the original columns after transformation.
    copy : bool, default=False
        Whether to copy the dataframe before transformation.
    verbose : bool, default=False
        Whether to print verbose output.
    return_native : bool, default=True
        Whether to return native dataframe (pandas/polars) or narwhals frame.

    Attributes
    ----------
    method : str
        Name of the pandas DataFrame method to apply.
    method_args : dict
        Arguments to pass to the method.
    columns : list of str
        Columns to transform.
    drop_original : bool
        Whether to drop the original columns after transformation.

    .. deprecated:: 0.5.0
        This transformer is deprecated and will be removed in a future version.
        Please use specific transformers for your use case.

    Examples
    --------
    >>> from tubular.base import DataFrameMethodTransformer
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> transformer = DataFrameMethodTransformer(method='fillna', method_args={'value': 0})
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       a  b
    0  1  4
    1  2  5
    2  3  6

    """

    # class attribute, indicates whether transformer works with lazyframes
    WORKS_WITH_LAZYFRAMES = False

    # class attribute, indicates whether transform requires fit to be run first
    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        method: str,
        method_args: Optional[dict[str, Any]] = None,
        columns: Optional[Union[str, List[str]]] = None,
        drop_original: bool = False,
        copy: bool = False,
        verbose: bool = False,
        return_native: bool = True,
    ):
        """Initialize DataFrameMethodTransformer.

        Parameters
        ----------
        method : str
            Name of the pandas DataFrame method to apply.
        method_args : dict, optional
            Arguments to pass to the method.
        columns : str or list of str, optional
            Columns to transform. If None, all columns are used.
        drop_original : bool, default=False
            Whether to drop the original columns after transformation.
        copy : bool, default=False
            Whether to copy the dataframe before transformation.
        verbose : bool, default=False
            Whether to print verbose output.
        return_native : bool, default=True
            Whether to return native dataframe (pandas/polars) or narwhals frame.

        """
        super().__init__(columns=columns, copy=copy, verbose=verbose, return_native=return_native)
        self.method = method
        self.method_args = method_args or {}
        self.drop_original = drop_original

    @beartype
    def transform(
        self,
        X: Annotated[
            nw.TypedFrame,
            Is[lambda frame: nw.is_dataframe(frame) and nw.len(frame) > 0],
        ],
    ) -> nw.TypedFrame:
        """Transform the dataframe by applying the specified method.

        Parameters
        ----------
        X : narwhals Frame
            Input dataframe to transform.

        Returns
        -------
        narwhals Frame
            Transformed dataframe.

        """
        X = super().transform(X, return_native_override=False)

        # Convert to pandas for method application
        if not isinstance(X, pd.DataFrame):
            X = X.to_pandas()

        # Apply the method
        method_func = getattr(X, self.method)
        X = method_func(**self.method_args)

        # Convert back to narwhals
        X = _convert_dataframe_to_narwhals(X)

        # Drop original columns if requested
        if self.drop_original and self.columns:
            X = X.drop(self.columns)

        return _return_narwhals_or_native_dataframe(X, self.return_native)
