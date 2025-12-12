"""Contains transformer for comparing equality between given columns (deprecated)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import narwhals as nw

from beartype import beartype
from typing_extensions import deprecated

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin
from tubular.types import (
    DataFrame,
    ListOfOneStr,
    ListOfTwoStrs,
    ListOfMoreThanOneStrings,
)

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)

if TYPE_CHECKING:
    import pandas as pd


class WhenThenOtherwiseTransformer(BaseTransformer):
    """Transformer to apply conditional logic across multiple columns.

    This transformer evaluates specified columns within a DataFrame and applies
    conditions based on logical rules to update column values.

    Attributes
    ----------
    polars_compatible : bool
        Indicates whether transformer has been converted to polars/pandas agnostic narwhals framework.

    FITS : bool
        Indicates whether transform requires fit to be run first.

    jsonable : bool
        Indicates if transformer supports to/from_json methods.

    lazyframe_compatible : bool
        Indicates whether transformer works with lazyframes.

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> from tubular.base import BaseTransformer

    >>> df = pd.DataFrame({
    ...     'col1': [1, 2, 3],
    ...     'col2': [4, 5, 6],
    ...     'condition_col': [True, False, True],
    ...     'update_col': [10, 20, 30]
    ... })

    >>> transformer = WhenThenOtherwiseTransformer(
    ...     columns=['col1', 'col2'],
    ...     condition_column='condition_col',
    ...     updated_values_column='update_col'
    ... )

    >>> transformer.transform(df)
       col1  col2  condition_col  update_col
    0    10    10          True         10
    1     2     5         False         20
    2    30    30          True         30

    ```
    """


    polars_compatible = True
    FITS = False
    jsonable = True
    lazyframe_compatible = True

    @beartype
    def __init__(
        self, 
        columns: ListOfMoreThanOneStrings, 
        condition_column: bool, 
        updated_values_column: ListOfOneStr, 
        **kwargs: dict[str, bool]) -> None:
        """
        Initialize the WhenThenOtherwiseTransformer.

        Parameters
        ----------
        columns : ListOfMoreThanOneStrings
            List of columns to be transformed.

        condition_column : bool
            Boolean column used to evaluate conditions.

        updated_values_column : ListOfOneStr
            Column containing values to update the specified columns based on the condition.

        **kwargs : dict[str, bool]
            Additional keyword arguments passed to the BaseTransformer.
        """
        
        super().__init__(columns=columns, **kwargs)


        self.condition_column = condition_column
        self.updated_values_column = updated_values_column

    @beartype
    def transform(
        self, 
        X: DataFrame,
        ) -> DataFrame:
        """Apply conditional logic to transform specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with updated columns based on conditions.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'col1': [1, 2, 3],
        ...     'col2': [4, 5, 6],
        ...     'condition_col': [True, False, True],
        ...     'update_col': [10, 20, 30]
        ... })

        >>> transformer = WhenThenOtherwiseTransformer(
        ...     columns=['col1', 'col2'],
        ...     condition_column='condition_col',
        ...     updated_values_column='update_col'
        ... )

        >>> transformer.transform(df)
           col1  col2  condition_col  update_col
        0    10    10          True         10
        1     2     5         False         20
        2    30    30          True         30

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        expr = {}
        for col in self.columns:
            expr[col] = nw.when(nw.col(self.condition_column)).then(nw.col(self.updated_values_column)).otherwise(nw.col(col))
        
        X = X.with_columns(*expr.values())

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class CompareTwoColumnsTransformer(BaseTransformer):
    """Transformer to compare two columns and generate outcomes based on conditions.

    This transformer evaluates a condition between two columns and generates an
    outcome based on the result.

    Attributes
    ----------
    polars_compatible : bool
        Indicates whether transformer has been converted to polars/pandas agnostic narwhals framework.

    FITS : bool
        Indicates whether transform requires fit to be run first.

    jsonable : bool
        Indicates if transformer supports to/from_json methods.

    lazyframe_compatible : bool
        Indicates whether transformer works with lazyframes.

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> from tubular.base import BaseTransformer

    >>> df = pd.DataFrame({
    ...     'col1': [1, 2, 3],
    ...     'col2': [3, 2, 1]
    ... })

    >>> transformer = CompareTwoColumnsTransformer(
    ...     columns=['col1', 'col2'],
    ...     condition='row["col1"] > row["col2"]',
    ...     outcome='comparison_result'
    ... )

    >>> transformer.transform(df)
       col1  col2  comparison_result
    0     1     3                  0
    1     2     2                  0
    2     3     1                  1

    ```

    """

    polars_compatible = True
    FITS = False
    jsonable = True
    lazyframe_compatible = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs, 
        condition: str, 
        outcome: str, 
        **kwargs: dict[str, bool]) -> None:
        """
        Initialize the CompareTwoColumnsTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            Tuple or list containing the names of the two columns to be compared.

        condition : str
            Logical condition to evaluate the relationship between the two columns.

        outcome : str
            Specification of the transformation or encoding to apply based on the evaluated condition.

        **kwargs : dict[str, bool]
            Additional keyword arguments passed to the BaseTransformer.
        """
        super().__init__(columns=columns, **kwargs)
        self.condition = condition
        self.outcome = outcome

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform two columns based on a condition to generate an outcome.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with the new outcome column.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'col1': [1, 2, 3],
        ...     'col2': [3, 2, 1]
        ... })

        >>> transformer = CompareTwoColumnsTransformer(
        ...     columns=['col1', 'col2'],
        ...     condition='row["col1"] > row["col2"]',
        ...     outcome='comparison_result'
        ... )

        >>> transformer.transform(df)
           col1  col2  comparison_result
        0     1     3                  0
        1     2     2                  0
        2     3     1                  1
        ```
        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        expr = nw.when(nw.expr(self.condition)).then(1).otherwise(0)

        X = X.with_columns(expr.alias(self.outcome))

        return _return_narwhals_or_native_dataframe(X, self.return_native)


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

    Attributes
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json,
        which limits it's supported functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to
        polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    deprecated: bool
        indicates if class has been deprecated

    """

    polars_compatible = False

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    deprecated = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        new_column_name: str,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialise class instance.

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

        """
        super().__init__(columns=columns, **kwargs)

        self.drop_original = drop_original
        self.new_column_name = new_column_name

    def get_feature_names_out(self) -> list[str]:
        """Get list of features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> # base classes just return inputs
        >>> transformer = EqualityChecker(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> transformer.get_feature_names_out()
        ['bla']

        ```

        """
        return [self.new_column_name]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create a column which indicated equality between given columns.

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
            X,
            self.drop_original,
            self.columns,
        )


