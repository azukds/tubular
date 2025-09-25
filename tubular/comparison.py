from __future__ import annotations

from typing import Optional

import pandas as pd  # noqa: TCH002
from beartype import beartype
from typing_extensions import deprecated

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin
from tubular.types import ListOfTwoStrs  # noqa: TCH001


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If it is useful to you, please raise an issue
    for it to be modernised
    """,
)
class EqualityChecker(
    DropOriginalMixin,
    BaseTransformer,
):
    """Transformer to check if two columns are equal.

    Parameters
    ----------
    columns: list
        List containing names of the two columns to check.

    new_column_name: string
        string containing the name of the new column.

    drop_original: boolean = False
        boolean representing dropping the input columns from X after checks.

    **kwargs:
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    """

    polars_compatible = False

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        new_column_name: str,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        self.drop_original = drop_original
        self.new_column_name = new_column_name

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create a column which is populated by the boolean
        matching between two columns iterated over rows.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with additional boolean column.

        """
        X = super().transform(X)

        X[self.new_column_name] = X[self.columns[0]] == X[self.columns[1]]

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
        )
