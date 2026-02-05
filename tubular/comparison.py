"""module for comparing and conditionally updating provided columns."""

from __future__ import annotations

import operator
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Optional

import narwhals as nw

if TYPE_CHECKING:
    import pandas as pd
from beartype import beartype
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin
from tubular.types import (
    DataFrame,
    Is,
    ListOfTwoStrs,
    NonEmptyListOfStrs,
    NumericTypes,
)


class ConditionEnum(Enum):
    """Enumeration of comparison conditions."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL_TO = "=="
    NOT_EQUAL_TO = "!="


ConditionEnumStr = Annotated[
    str,
    Is[lambda s: s in ConditionEnum._value2member_map_],
]


class WhenThenOtherwiseTransformer(BaseTransformer):
    """Transformer to apply conditional logic across multiple columns.

    This transformer evaluates specified columns against a condition and updates
    with given values based on the results.

    Attributes
    ----------
    polars_compatible : bool
        Indicates whether transformer has been converted to polars/pandas agnostic
        narwhals framework.

    FITS : bool
        Indicates whether transform requires fit to be run first.

    jsonable : bool
        Indicates if transformer supports to/from_json methods.

    lazyframe_compatible : bool
        Indicates whether transformer works with lazyframes.

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [4, 5, 6],
    ...         "condition_col": [True, False, True],
    ...         "update_col": [10, 20, 30],
    ...     }
    ... )
    >>> transformer = WhenThenOtherwiseTransformer(
    ...     columns=["a", "b"], when_column="condition_col", then_column="update_col"
    ... )
    >>> transformed_df = transformer.transform(df)
    >>> print(transformed_df)
    shape: (3, 4)
    ┌─────┬─────┬───────────────┬────────────┐
    │ a   ┆ b   ┆ condition_col ┆ update_col │
    │ --- ┆ --- ┆ ---           ┆ ---        │
    │ i64 ┆ i64 ┆ bool          ┆ i64        │
    ╞═════╪═════╪═══════════════╪════════════╡
    │ 10  ┆ 10  ┆ true          ┆ 10         │
    │ 2   ┆ 5   ┆ false         ┆ 20         │
    │ 30  ┆ 30  ┆ true          ┆ 30         │
    └─────┴─────┴───────────────┴────────────┘

    ```

    """

    polars_compatible = True
    FITS = False
    jsonable = True
    lazyframe_compatible = True

    @beartype
    def __init__(
        self,
        columns: NonEmptyListOfStrs,
        when_column: str,
        then_column: str,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialize the WhenThenOtherwiseTransformer.

        Parameters
        ----------
        columns : ListOfMoreThanOneStrings
            List of columns to be transformed.

        when_column : bool
            Boolean column used to evaluate conditions.

        then_column : ListOfOneStr
            Column containing values to update the specified columns
            based on the condition.

        **kwargs : dict[str, bool]
            Additional keyword arguments passed to the BaseTransformer.

        """
        super().__init__(columns=columns, **kwargs)

        self.when_column = when_column
        self.then_column = then_column

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
        >>> from pprint import pprint
        >>> transformer = WhenThenOtherwiseTransformer(
        ...     columns=["a", "b"],
        ...     when_column="condition_col",
        ...     then_column="update_col",  # noqa: E501
        ... )
        >>> pprint(transformer.to_json(), sort_dicts=True)
        {'classname': 'WhenThenOtherwiseTransformer',
         'fit': {},
         'init': {'columns': ['a', 'b'],
                  'copy': False,
                  'return_native': True,
                  'then_column': 'update_col',
                  'verbose': False,
                  'when_column': 'condition_col'},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "when_column": self.when_column,
                "then_column": self.then_column,
            },
        )

        return json_dict

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Apply conditional logic to transform specified columns.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to be transformed.

        Returns
        -------
        DataFrame
            Transformed DataFrame with updated columns based on conditions.

        Raises
        ------
        TypeError
            If the `when_column` is not of type Boolean or if columns
            have mismatched types.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [4, 5, 6],
        ...         "condition_col": [True, False, True],
        ...         "update_col": [10, 20, 30],
        ...     }
        ... )
        >>> transformer = WhenThenOtherwiseTransformer(
        ...     columns=["a", "b"],
        ...     when_column="condition_col",
        ...     then_column="update_col",
        ... )
        >>> transformed_df = transformer.transform(df)
        >>> print(transformed_df)
        shape: (3, 4)
        ┌─────┬─────┬───────────────┬────────────┐
        │ a   ┆ b   ┆ condition_col ┆ update_col │
        │ --- ┆ --- ┆ ---           ┆ ---        │
        │ i64 ┆ i64 ┆ bool          ┆ i64        │
        ╞═════╪═════╪═══════════════╪════════════╡
        │ 10  ┆ 10  ┆ true          ┆ 10         │
        │ 2   ┆ 5   ┆ false         ┆ 20         │
        │ 30  ┆ 30  ┆ true          ┆ 30         │
        └─────┴─────┴───────────────┴────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        schema = X.schema
        if schema[self.when_column] != nw.Boolean:
            message = f"The column '{self.when_column}' must be of type Boolean."
            raise TypeError(message)

        then_column_type = schema[self.then_column]
        if any(schema[col] != then_column_type for col in self.columns):
            message = (
                f"All columns in {self.columns} must be of the same type as "
                f"'{self.then_column}'."
            )
            raise TypeError(message)

        exprs_dict = {
            col: nw.when(nw.col(self.when_column))
            .then(nw.col(self.then_column))
            .otherwise(nw.col(col))
            for col in self.columns
        }

        X = X.with_columns(**exprs_dict)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class CompareTwoColumnsTransformer(BaseTransformer):
    """Transformer to compare two columns and generate outcomes based on conditions.

    This transformer evaluates a condition between two columns and generates an
    outcome based on the result.

    Attributes
    ----------
    polars_compatible : bool
        Indicates whether transformer has been converted to polars/pandas
        agnostic narwhals framework.

    FITS : bool
        Indicates whether transform requires fit to be run first.

    jsonable : bool
        Indicates if transformer supports to/from_json methods.

    lazyframe_compatible : bool
        Indicates whether transformer works with lazyframes.

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    >>> transformer = CompareTwoColumnsTransformer(
    ...     columns=["a", "b"],
    ...     condition=">",
    ... )
    >>> transformed_df = transformer.transform(df)
    >>> print(transformed_df)
    shape: (3, 3)
    ┌─────┬─────┬───────┐
    │ a   ┆ b   ┆ a>b   │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ i64 ┆ bool  │
    ╞═════╪═════╪═══════╡
    │ 1   ┆ 3   ┆ false │
    │ 2   ┆ 2   ┆ false │
    │ 3   ┆ 1   ┆ true  │
    └─────┴─────┴───────┘

    ```

    """

    polars_compatible = True
    FITS = False
    jsonable = True
    lazyframe_compatible = True

    # Map the enum to the operator functions
    ops_map: ClassVar[dict[ConditionEnum, Any]] = {
        ConditionEnum.GREATER_THAN: operator.gt,
        ConditionEnum.LESS_THAN: operator.lt,
        ConditionEnum.EQUAL_TO: operator.eq,
        ConditionEnum.NOT_EQUAL_TO: operator.ne,
    }

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        condition: ConditionEnumStr,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialize the CompareTwoColumnsTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            Tuple or list containing the names of the two columns to be compared.

        condition : str
            Logical condition to evaluate the relationship between the two columns.

        **kwargs : dict[str, bool]
            Additional keyword arguments passed to the BaseTransformer.

        """
        super().__init__(columns=columns, **kwargs)
        self.condition = ConditionEnum(condition)

    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        ```pycon
        >>> transformer = CompareTwoColumnsTransformer(
        ...     columns=["a", "b"],
        ...     condition=ConditionEnum.GREATER_THAN.value,
        ... )
        >>> json_dict = transformer.to_json()
        >>> from pprint import pprint
        >>> pprint(json_dict, sort_dicts=True)
        {'classname': 'CompareTwoColumnsTransformer',
         'fit': {},
         'init': {'columns': ['a', 'b'],
                  'condition': '>',
                  'copy': False,
                  'return_native': True,
                  'verbose': False},
         'tubular_version': ...}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["condition"] = self.condition.value

        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform two columns based on a condition to generate an outcome.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to be transformed.

        Returns
        -------
        DataFrame
            Transformed DataFrame with the new outcome column.

        Raises
        ------
        TypeError
            If the columns are not of a numeric type.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
        >>> transformer = CompareTwoColumnsTransformer(
        ...     columns=["a", "b"],
        ...     condition=">",
        ... )
        >>> transformed_df = transformer.transform(df)
        >>> print(transformed_df)
        shape: (3, 3)
        ┌─────┬─────┬───────┐
        │ a   ┆ b   ┆ a>b   │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1   ┆ 3   ┆ false │
        │ 2   ┆ 2   ┆ false │
        │ 3   ┆ 1   ┆ true  │
        └─────┴─────┴───────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        schema = X.schema

        bad_cols = [col for col in self.columns if schema[col] not in NumericTypes]
        if bad_cols:
            message = (
                "Columns must be of a numeric type, but the following are not: "
                f"{bad_cols}"
            )
            raise TypeError(message)

        null_filter_expr = (
            nw.col(self.columns[0]).is_null() | nw.col(self.columns[1]).is_null()
        )

        expr = (
            nw.when(~null_filter_expr)
            .then(
                self.ops_map[self.condition](
                    nw.col(self.columns[0]), nw.col(self.columns[1])
                )
            )
            .otherwise(None)
        )

        backend = nw.get_native_namespace(X).__name__

        if backend == "polars":
            expr = expr.cast(nw.Boolean)

        outcome_column_name = (
            f"{self.columns[0]}{self.condition.value}{self.columns[1]}"
        )

        X = X.with_columns(expr.alias(outcome_column_name))

        if backend == "pandas":
            X = X.with_columns(
                nw.maybe_convert_dtypes(X[outcome_column_name]).cast(nw.Boolean)
            )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


# DEPRECATED TRANSFORMERS
@deprecated(
    """This transformer has been superseded by CompareTwoColumnsTransformer
    and so has been deprecated, and will be removed in a future major release.
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
