from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import pandas as pd

if TYPE_CHECKING:
    from narwhals.typing import FrameT

from tubular.base import BaseTransformer


class SetValueTransformer(BaseTransformer):
    """Transformer to set value of column(s) to a given value.

    This should be used if columns need to be set to a constant value.

    Parameters
    ----------
    columns: list or str
        Columns to set values.

    value : various
        Value to set.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = True

    def __init__(
        self,
        columns: str | list[str],
        value: type,
        copy: bool = False,
        **kwargs: dict[str, bool],
    ) -> None:
        self.value = value

        super().__init__(columns=columns, copy=copy, **kwargs)

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Set columns to value.

        Parameters
        ----------
        X : FrameT
            Data to apply mappings to.

        Returns
        -------
        X : FrameT
            Transformed input X with columns set to value.

        """
        X = nw.from_native(super().transform(X))

        set_value_expression = [nw.lit(self.value).alias(col) for col in self.columns]
        return X.with_columns(set_value_expression)


class ColumnDtypeSetter(BaseTransformer):
    """Transformer to set transform columns in a dataframe to a dtype.

    Parameters
    ----------
    columns : str or list
        Columns to set dtype. Must be set or transform will not run.

    dtype : type or string
        dtype object to set columns to or a string interpretable as one by pd.api.types.pandas_dtype
        e.g. float or 'float'

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework
    """

    polars_compatible = False

    def __init__(
        self,
        columns: str | list[str],
        dtype: type | str,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns, **kwargs)

        self.__validate_dtype(dtype)

        self.dtype = dtype

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        X[self.columns] = X[self.columns].astype(self.dtype)

        return X

    def __validate_dtype(self, dtype: str) -> None:
        """Check string is a valid dtype."""
        try:
            pd.api.types.pandas_dtype(dtype)
        except TypeError:
            msg = f"{self.classname()}: data type '{dtype}' not understood as a valid dtype"
            raise TypeError(msg) from None
