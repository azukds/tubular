"""Contains transformers for working with date columns."""

from __future__ import annotations

import copy
import datetime
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.vale import Is
from typing_extensions import deprecated

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.mixins import DropOriginalMixin
from tubular.types import (
    DataFrame,
    GenericKwargs,
    ListOfOneStr,
    ListOfThreeStrs,
    ListOfTwoStrs,
)

if TYPE_CHECKING:
    from narwhals.typing import FrameT

TIME_UNITS = ["us", "ns", "ms"]


@register
class BaseGenericDateTransformer(
    DropOriginalMixin,
    BaseTransformer,
):
    """Extends BaseTransformer for datetime/date scenarios.

    Attributes:
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> BaseGenericDateTransformer(
    ...     columns=["a", "b"],
    ...     new_column_name="bla",
    ... )
    BaseGenericDateTransformer(columns=['a', 'b'], new_column_name='bla')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[list[str], str],
        new_column_name: str,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : Union[list[str], str]
            List of 2 columns. First column will be subtracted from second.

        new_column_name : str
            Name for the new year column.

        drop_original : bool
            Flag for whether to drop the original columns.

        return_native: bool, default = True
            Controls whether transformer returns narwhals or native pandas/polars type

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

        self.drop_original = drop_original
        self.new_column_name = new_column_name

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> transformer = BaseGenericDateTransformer(columns=["a", "b"], new_column_name="bla")

        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'BaseGenericDateTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'bla', 'drop_original': False}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"]["new_column_name"] = self.new_column_name
        json_dict["init"]["drop_original"] = self.drop_original

        return json_dict

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> # base classes just return inputs
        >>> transformer = BaseGenericDateTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> transformer.get_feature_names_out()
        ['a', 'b']

        >>> # other classes return new columns
        >>> transformer = DateDifferenceTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> transformer.get_feature_names_out()
        ['bla']

        ```

        """
        # base classes just return columns, so need special handling
        return (
            [*self.columns]
            if type(self)
            in {
                BaseGenericDateTransformer,
                BaseDatetimeTransformer,
            }
            else [self.new_column_name]
        )

    @beartype
    def check_columns_are_date_or_datetime(
        self,
        X: DataFrame,
        datetime_only: bool,
    ) -> None:
        """Check types of provided columns.

        Columns must be datetime or date type, depending on the datetime_only
        flag. If a column does not meet the expected type criteria, a TypeError is raised.

        Parameters
        ----------
        X: DataFrame
            Data to validate

        datetime_only: bool
            Indicates whether ONLY datetime types are accepted

        Raises
        ------
        TypeError: if non date/datetime types are found

        TypeError: if mismatched date/datetime types are found,
        types should be consistent

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseGenericDateTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.date(1993, 9, 27), datetime.date(2005, 10, 7)],
        ...         "b": [datetime.date(1991, 5, 22), datetime.date(2001, 12, 10)],
        ...     },
        ... )

        >>> transformer.check_columns_are_date_or_datetime(test_df, datetime_only=False)

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        type_msg = ["Datetime"]
        date_type = nw.Date
        allowed_types = [nw.Datetime]
        if not datetime_only:
            allowed_types = [*allowed_types, date_type]
            type_msg += ["Date"]

        schema = X.schema

        for col in self.columns:
            is_datetime = False
            is_date = False
            if isinstance(schema[col], nw.Datetime):
                is_datetime = True

            elif schema[col] == nw.Date:
                is_date = True

            # first check for invalid types (non date/datetime)
            if (not is_datetime) and (not (not datetime_only and is_date)):
                msg = f"{self.classname()}: {col} type should be in {type_msg} but got {schema[col]}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones()"
                raise TypeError(msg)

        # process datetime types for more readable error messages
        present_types = {
            dtype if not isinstance(dtype, nw.Datetime) else nw.Datetime
            for name, dtype in schema.items()
            if name in self.columns
        }

        valid_types = present_types.issubset(set(allowed_types))
        # convert to list and sort to ensure reproducible order
        present_types = {str(value) for value in present_types}
        present_types = list(present_types)
        present_types.sort()

        # next check for consistent types (all date or all datetime)
        if not valid_types or len(present_types) > 1:
            msg = rf"{self.classname()}: Columns fed to datetime transformers should be {type_msg} and have consistent types, but found {present_types}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones(). Please use ToDatetimeTransformer to standardise."
            raise TypeError(
                msg,
            )

    @beartype
    def transform(
        self,
        X: DataFrame,
        datetime_only: bool = False,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Validate data pre transform.

        Parameters
        ----------
        X : DataFrame
            Data containing self.columns

        datetime_only: bool
            Indicates whether ONLY datetime types are accepted

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Validated data

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = BaseGenericDateTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.date(1993, 9, 27), datetime.date(2005, 10, 7)],
        ...         "b": [datetime.date(1991, 5, 22), datetime.date(2001, 12, 10)],
        ...     },
        ... )

        >>> # base transform has no effect on data
        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌────────────┬────────────┐
        │ a          ┆ b          │
        │ ---        ┆ ---        │
        │ date       ┆ date       │
        ╞════════════╪════════════╡
        │ 1993-09-27 ┆ 1991-05-22 │
        │ 2005-10-07 ┆ 2001-12-10 │
        └────────────┴────────────┘

        ```

        """
        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        X = _convert_dataframe_to_narwhals(X)

        self.check_columns_are_date_or_datetime(X, datetime_only=datetime_only)

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class BaseDatetimeTransformer(BaseGenericDateTransformer):
    """Extends BaseTransformer for datetime scenarios.

    Attributes:
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> BaseDatetimeTransformer(
    ...     columns=["a", "b"],
    ...     new_column_name="bla",
    ... )
    BaseDatetimeTransformer(columns=['a', 'b'], new_column_name='bla')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = False

    @beartype
    def __init__(
        self,
        columns: Union[list[str], str],
        new_column_name: str,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : Union[list[str], str]
            List of 2 columns. First column will be subtracted from second.

        new_column_name : str
            Name for the new year column.

        drop_original : bool
            Flag for whether to drop the original columns.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(
            columns=columns,
            new_column_name=new_column_name,
            drop_original=drop_original,
            **kwargs,
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Check types of selected columns in provided data.

        Parameters
        ----------
        X : DataFrame
            Data containing self.columns

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Validated data

        Example:
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = BaseDatetimeTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="bla",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.datetime(1993, 9, 27), datetime.datetime(2005, 10, 7)],
        ...         "b": [datetime.datetime(1991, 5, 22), datetime.datetime(2001, 12, 10)],
        ...     },
        ... )

        >>> # base transform has no effect on data
        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────────────────────┬─────────────────────┐
        │ a                   ┆ b                   │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 1993-09-27 00:00:00 ┆ 1991-05-22 00:00:00 │
        │ 2005-10-07 00:00:00 ┆ 2001-12-10 00:00:00 │
        └─────────────────────┴─────────────────────┘

        ```

        """
        return_native = self._process_return_native(return_native_override)

        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, datetime_only=True, return_native_override=False)

        return _return_narwhals_or_native_dataframe(X, return_native)


class DateDifferenceUnitsOptions(str, Enum):
    """Options for return units in DateDifferenceTransformer."""

    __slots__ = ()

    WEEK = "week"
    FORTNIGHT = "fortnight"
    LUNAR_MONTH = "lunar_month"
    COMMON_YEAR = "common_year"
    CUSTOM_DAYS = "custom_days"
    DAYS = "D"
    HOURS = "h"
    MINUTES = "m"
    SECONDS = "s"


DateDifferenceUnitsOptionsStr = Annotated[
    str,
    Is[lambda s: s in DateDifferenceUnitsOptions._value2member_map_],
]


@register
class DateDifferenceTransformer(BaseGenericDateTransformer):
    """Class to transform calculate the difference between 2 date fields in specified units.

    Attributes:
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> transformer = DateDifferenceTransformer(
    ...     columns=["a", "b"],
    ...     new_column_name="bla",
    ...     units="common_year",
    ... )
    >>> transformer
    DateDifferenceTransformer(columns=['a', 'b'], new_column_name='bla',
                              units='common_year')

    >>> # transformer can also be dumped to json and reinitialised

    >>> json_dump = transformer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'DateDifferenceTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'bla', 'drop_original': False, 'units': 'common_year', 'custom_days_divider': None}, 'fit': {}}

    >>> DateDifferenceTransformer.from_json(json_dump)
    DateDifferenceTransformer(columns=['a', 'b'], new_column_name='bla',
                              units='common_year')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        new_column_name: str,
        units: DateDifferenceUnitsOptionsStr = "D",
        drop_original: bool = False,
        custom_days_divider: Optional[int] = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : List[str]
            List of 2 columns. First column will be subtracted from second.
        new_column_name : str, default = None
            Name given to calculated datediff column. If None then {column_upper}_{column_lower}_datediff_{units}
            will be used.
        units : str, default = 'D'
            Accepted values are "week", "fortnight", "lunar_month", "common_year", "custom_days", 'D', 'h', 'm', 's'
        copy : bool, default = False
            Should X be copied prior to transform? Copy argument no longer used and will be deprecated in a future release
        verbose: bool, default = False
            Control level of detail in printouts
        drop_original:
            Boolean flag indicating whether to drop original columns.
        custom_days_divider:
            Integer value for the "custom_days" unit
        kwargs:
            arguments for base class, e.g. verbose

        """
        self.units = units
        self.custom_days_divider = custom_days_divider

        super().__init__(
            columns=columns,
            new_column_name=new_column_name,
            drop_original=drop_original,
            **kwargs,
        )

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = columns[0]
        self.column_upper = columns[1]

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> transformer = DateDifferenceTransformer(columns=["a", "b"], new_column_name="a_diff_b")

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'DateDifferenceTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'a_diff_b', 'drop_original': False, 'units': 'D', 'custom_days_divider': None}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "new_column_name": self.new_column_name,
                "units": self.units,
                "drop_original": self.drop_original,
                "custom_days_divider": self.custom_days_divider,
            },
        )

        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Calculate the difference between the given fields in the specified units.

        Parameters
        ----------
        X : DataFrame
            Data containing self.columns

        Returns
        -------
        DataFrame:
            dataframe with added date difference column

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = DateDifferenceTransformer(
        ...     columns=["a", "b"],
        ...     new_column_name="a_b_difference_years",
        ...     units="common_year",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.date(1993, 9, 27), datetime.date(2005, 10, 7)],
        ...         "b": [datetime.date(1991, 5, 22), datetime.date(2001, 12, 10)],
        ...     },
        ... )

        >>> transformer.transform(test_df)
        shape: (2, 3)
        ┌────────────┬────────────┬──────────────────────┐
        │ a          ┆ b          ┆ a_b_difference_years │
        │ ---        ┆ ---        ┆ ---                  │
        │ date       ┆ date       ┆ f64                  │
        ╞════════════╪════════════╪══════════════════════╡
        │ 1993-09-27 ┆ 1991-05-22 ┆ -2.353425            │
        │ 2005-10-07 ┆ 2001-12-10 ┆ -3.827397            │
        └────────────┴────────────┴──────────────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        # mapping for units and corresponding timedelta arg values
        UNITS_TO_TIMEDELTA_PARAMS = {
            "week": (7, "D"),
            "fortnight": (14, "D"),
            "lunar_month": (
                int(29.5 * 24),
                "h",
            ),  # timedelta values need to be whole numbers so (29.5, 'D') cannot be used
            "common_year": (365, "D"),
            "D": (1, "D"),
            "h": (1, "h"),
            "m": (1, "m"),
            "s": (1, "s"),
        }

        # list of units that require time truncation
        UNITS_TO_TRUNCATE_TIME_FOR = [
            "week",
            "fortnight",
            "lunar_month",
            "common_year",
            "custom_days",
            "D",
        ]

        start_date_col = nw.col(self.columns[0])
        end_date_col = nw.col(self.columns[1])

        # truncating time for specific units
        if self.units in UNITS_TO_TRUNCATE_TIME_FOR:
            start_date_col = start_date_col.dt.truncate("1d")
            end_date_col = end_date_col.dt.truncate("1d")

        if self.units == "custom_days":
            timedelta_value, timedelta_format = self.custom_days_divider, "D"
            denominator = np.timedelta64(timedelta_value, timedelta_format)
        else:
            timedelta_value, timedelta_format = UNITS_TO_TIMEDELTA_PARAMS[self.units]
            denominator = np.timedelta64(timedelta_value, timedelta_format)

        X = X.with_columns(
            ((end_date_col - start_date_col) / denominator).alias(self.new_column_name),
        )

        # Drop original columns if self.drop_original is True
        X = DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class ToDatetimeTransformer(BaseTransformer):
    """Class to transform convert specified columns to datetime.

    Class simply uses the pd.to_datetime method on the specified columns.

    Attributes:
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> transformer = ToDatetimeTransformer(
    ...     columns="a",
    ...     time_format="%d/%m/%Y",
    ... )
    >>> transformer
    ToDatetimeTransformer(columns=['a'], time_format='%d/%m/%Y')

    >>> # version will vary for local vs CI, so use ... as generic match
    >>> json_dump = transformer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ToDatetimeTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'time_format': '%d/%m/%Y'}, 'fit': {}}

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        time_format: Optional[str] = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : List[str]
            List of names of the column to convert to datetime.

        time_format: str
            str indicating format of time to parse, e.g. '%d/%m/%Y'

        **kwargs
            Arbitrary keyword arguments passed onto pd.to_datetime().

        """
        if not time_format:
            warnings.warn(
                "time_format arg has not been provided, so datetime format will be inferred",
                stacklevel=2,
            )

        self.time_format = time_format

        super().__init__(
            columns=columns,
            **kwargs,
        )

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> transformer = ToDatetimeTransformer(columns="a", time_format="%d/%m/%Y")

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'ToDatetimeTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'time_format': '%d/%m/%Y'}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()
        json_dict["init"].update(
            {
                "time_format": self.time_format,
            }
        )
        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Convert specified column to datetime using pd.to_datetime.

        Parameters
        ----------
        X : DataFrame
            Data with column to transform.

        Returns
        -------
        DataFrame:
            dataframe with provided columns converted to datetime

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = ToDatetimeTransformer(
        ...     columns="a",
        ...     time_format="%d/%m/%Y",
        ... )

        >>> test_df = pl.DataFrame({"a": ["01/02/2020", "10/12/1996"], "b": [1, 2]})

        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────────────────────┬─────┐
        │ a                   ┆ b   │
        │ ---                 ┆ --- │
        │ datetime[μs]        ┆ i64 │
        ╞═════════════════════╪═════╡
        │ 2020-02-01 00:00:00 ┆ 1   │
        │ 1996-12-10 00:00:00 ┆ 2   │
        └─────────────────────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        X = X.with_columns(
            nw.col(col).str.to_datetime(format=self.time_format) for col in self.columns
        )

        return _return_narwhals_or_native_dataframe(X, return_native=self.return_native)


@register
class BetweenDatesTransformer(BaseGenericDateTransformer):
    """Transformer to generate a boolean column indicating if one date is between two others.

    If not all column_lower values are less than or equal to column_upper when transform is run
    then a warning will be raised.

    Attributes:
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    column_lower : str
        Name of date column to subtract. This attribute is not for use in any method,
        use 'columns' instead. Here only as a fix to allow string representation of transformer.

    column_upper : str
        Name of date column to subtract from. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    column_between : str
        Name of column to check if it's values fall between column_lower and column_upper. This attribute
        is not for use in any method, use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : list
        Contains the names of the columns to compare in the order [column_lower, column_between
        column_upper].

    new_column_name : str
        new_column_name argument passed when initialising the transformer.

    lower_inclusive : bool
        lower_inclusive argument passed when initialising the transformer.

    upper_inclusive : bool
        upper_inclusive argument passed when initialising the transformer.

    drop_original: bool
        indicates whether to drop original columns.

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> BetweenDatesTransformer(
    ...     columns=["a", "b", "c"],
    ...     new_column_name="b_between_a_c",
    ...     lower_inclusive=True,
    ...     upper_inclusive=True,
    ... )
    BetweenDatesTransformer(columns=['a', 'b', 'c'],
                            new_column_name='b_between_a_c')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: ListOfThreeStrs,
        new_column_name: str,
        drop_original: bool = False,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : list[str]
            List of columns for comparison, in format [lower, to_compare, upper]

        new_column_name : str
            Name for new column to be added to X.

        drop_original: bool
            indicates whether to drop original columns.

        lower_inclusive : bool, default = True
            If lower_inclusive is True the comparison to column_lower will be column_lower <=
            column_between, otherwise the comparison will be column_lower < column_between.

        upper_inclusive : bool, default = True
            If upper_inclusive is True the comparison to column_upper will be column_between <=
            column_upper, otherwise the comparison will be column_between < column_upper.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.__init__().

        """
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive

        super().__init__(
            columns=columns,
            new_column_name=new_column_name,
            drop_original=drop_original,
            **kwargs,
        )

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = columns[0]
        self.column_upper = columns[2]
        self.column_between = columns[2]

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> transformer = BetweenDatesTransformer(
        ...     columns=["a", "b", "c"],
        ...     new_column_name="b_between_a_c",
        ...     lower_inclusive=True,
        ...     upper_inclusive=False,
        ... )
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'BetweenDatesTransformer', 'init': {'columns': ['a', 'b', 'c'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'b_between_a_c', 'drop_original': False, 'lower_inclusive': True, 'upper_inclusive': False}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "lower_inclusive": self.lower_inclusive,
                "upper_inclusive": self.upper_inclusive,
            },
        )

        return json_dict

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Transform - creates column indicating if middle date is between the other two.

        If not all column_lower values are less than or equal to column_upper when transform is run
        then a warning will be raised.

        Parameters
        ----------
        X : pd/pl/nw.DataFrame
            Data to transform.

        Returns
        -------
        X : pd/pl/nw.DataFrame
            Input X with additional column (self.new_column_name) added. This column is
            boolean and indicates if the middle column is between the other 2.

        Example:
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = BetweenDatesTransformer(
        ...     columns=["a", "b", "c"],
        ...     new_column_name="b_between_a_c",
        ...     lower_inclusive=True,
        ...     upper_inclusive=True,
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.date(1990, 9, 27), datetime.date(2005, 10, 7)],
        ...         "b": [datetime.date(1991, 5, 22), datetime.date(2001, 12, 10)],
        ...         "c": [datetime.date(1993, 4, 20), datetime.date(2007, 11, 8)],
        ...     },
        ... )

        >>> transformer.transform(test_df)
        shape: (2, 4)
        ┌────────────┬────────────┬────────────┬───────────────┐
        │ a          ┆ b          ┆ c          ┆ b_between_a_c │
        │ ---        ┆ ---        ┆ ---        ┆ ---           │
        │ date       ┆ date       ┆ date       ┆ bool          │
        ╞════════════╪════════════╪════════════╪═══════════════╡
        │ 1990-09-27 ┆ 1991-05-22 ┆ 1993-04-20 ┆ true          │
        │ 2005-10-07 ┆ 2001-12-10 ┆ 2007-11-08 ┆ false         │
        └────────────┴────────────┴────────────┴───────────────┘

        ```

        """
        X = nw.from_native(super().transform(X))

        if not (
            X.select((nw.col(self.columns[0]) <= nw.col(self.columns[2])).all()).item()
        ):
            warnings.warn(
                f"{self.classname()}: not all {self.columns[2]} are greater than or equal to {self.columns[0]}",
                stacklevel=2,
            )

        lower_comparison = (
            nw.col(self.columns[0]) <= nw.col(self.columns[1])
            if self.lower_inclusive
            else nw.col(self.columns[0]) < nw.col(self.columns[1])
        )

        upper_comparison = (
            nw.col(self.columns[1]) <= nw.col(self.columns[2])
            if self.upper_inclusive
            else nw.col(self.columns[1]) < nw.col(self.columns[2])
        )

        X = X.with_columns(
            (lower_comparison & upper_comparison).alias(self.new_column_name),
        )

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
        )


class DatetimeInfoOptions(str, Enum):
    """Options for what is returned by DatetimeInfoExtractor."""

    __slots__ = ()

    TIME_OF_DAY = "timeofday"
    TIME_OF_MONTH = "timeofmonth"
    TIME_OF_YEAR = "timeofyear"
    DAY_OF_WEEK = "dayofweek"


DatetimeInfoOptionStr = Annotated[
    str,
    Is[lambda s: s in DatetimeInfoOptions._value2member_map_],
]
DatetimeInfoOptionList = Annotated[
    list,
    Is[
        lambda list_value: all(
            entry in DatetimeInfoOptions._value2member_map_ for entry in list_value
        )
    ],
]


@register
class DatetimeInfoExtractor(BaseDatetimeTransformer):
    """Transformer to extract various features from datetime var.

    Attributes:
    ----------
    columns: List[str]
        List of columns for processing

    include : list of str, default = ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]
        Which datetime categorical information to extract

    datetime_mappings : dict, default = None
        Optional argument to define custom mappings for datetime values.

    drop_original: str
        indicates whether to drop provided columns post transform

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> transformer = DatetimeInfoExtractor(
    ...     columns="a",
    ...     include="timeofday",
    ... )
    >>> transformer
    DatetimeInfoExtractor(columns=['a'], datetime_mappings={},
                          include=['timeofday'])

    >>> transformer.to_json()
    {'tubular_version': ..., 'classname': 'DatetimeInfoExtractor', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'dummy', 'drop_original': False, 'include': ['timeofday'], 'datetime_mappings': {}}, 'fit': {}}

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    DEFAULT_MAPPINGS: ClassVar[dict[str, dict[int, str]]] = {
        DatetimeInfoOptions.TIME_OF_DAY.value: {
            **dict.fromkeys(range(6), "night"),  # Midnight - 6am
            **dict.fromkeys(range(6, 12), "morning"),  # 6am - Noon
            **dict.fromkeys(range(12, 18), "afternoon"),  # Noon - 6pm
            **dict.fromkeys(range(18, 24), "evening"),  # 6pm - Midnight
        },
        DatetimeInfoOptions.TIME_OF_MONTH.value: {
            **dict.fromkeys(range(1, 11), "start"),
            **dict.fromkeys(range(11, 21), "middle"),
            **dict.fromkeys(range(21, 32), "end"),
        },
        DatetimeInfoOptions.TIME_OF_YEAR.value: {
            **dict.fromkeys(range(3, 6), "spring"),  # Mar, Apr, May
            **dict.fromkeys(range(6, 9), "summer"),  # Jun, Jul, Aug
            **dict.fromkeys(range(9, 12), "autumn"),  # Sep, Oct, Nov
            **dict.fromkeys([12, 1, 2], "winter"),  # Dec, Jan, Feb
        },
        DatetimeInfoOptions.DAY_OF_WEEK.value: {
            1: "monday",
            2: "tuesday",
            3: "wednesday",
            4: "thursday",
            5: "friday",
            6: "saturday",
            7: "sunday",
        },
    }

    INCLUDE_OPTIONS: ClassVar[list[str]] = list(DEFAULT_MAPPINGS.keys())

    RANGE_TO_MAP: ClassVar[dict[str, set[int]]] = {
        DatetimeInfoOptions.TIME_OF_DAY.value: set(range(24)),
        DatetimeInfoOptions.TIME_OF_MONTH.value: set(range(1, 32)),
        DatetimeInfoOptions.TIME_OF_YEAR.value: set(range(1, 13)),
        DatetimeInfoOptions.DAY_OF_WEEK.value: set(range(1, 8)),
    }

    DATETIME_ATTR: ClassVar[dict[str, str]] = {
        DatetimeInfoOptions.TIME_OF_DAY.value: "hour",
        DatetimeInfoOptions.TIME_OF_MONTH.value: "day",
        DatetimeInfoOptions.TIME_OF_YEAR.value: "month",
        DatetimeInfoOptions.DAY_OF_WEEK.value: "weekday",
    }

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        include: Optional[Union[DatetimeInfoOptionList, DatetimeInfoOptionStr]] = None,
        datetime_mappings: Optional[dict[DatetimeInfoOptionStr, dict[int, str]]] = None,
        drop_original: Optional[bool] = False,
        **kwargs: Union[str, bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list
            datetime columns to extract information from

        include : list of str, default = ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]
            Which datetime categorical information to extract

        datetime_mappings : dict, default = {}
            Optional argument to define custom mappings for datetime values.
            Keys of the dictionary must be contained in `include`.
            All possible values of each feature must be included in the mappings,
            ie, a mapping for `dayofweek` must include all values 1-7;
            datetime_mappings = {
                                "dayofweek": {
                                            **{i: "week" for i in range(1,6)},
                                            **{i: "week" for i in range(6,8)}
                                            }
                                }

            The required ranges for each mapping are:
                timeofday: 0-23
                timeofmonth: 1-31
                timeofyear: 1-12
                dayofweek: 1-7

            If an option is present in 'include' but no mappings are provided,
            then default values from cls.DEFAULT_MAPPINGS will be used for this
            option.

        drop_original: str
            indicates whether to drop provided columns post transform

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        if include is None:
            include = self.INCLUDE_OPTIONS

        if "new_column_name" in kwargs:
            warnings.warn(
                f"{self.classname()}: new_column_name argument is not used for this class",
                stacklevel=2,
            )

            kwargs.pop("new_column_name")

        super().__init__(
            columns=columns,
            drop_original=drop_original,
            new_column_name="dummy",
            **kwargs,
        )

        if isinstance(include, str):
            include = [include]

        self.include = include

        self._check_provided_mappings(datetime_mappings=datetime_mappings)

        self.datetime_mappings = datetime_mappings

        if self.datetime_mappings is None:
            self.datetime_mappings = {}

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        >>> transformer=DatetimeInfoExtractor(columns='a')

        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'DatetimeInfoExtractor', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'dummy', 'drop_original': False, 'include': ['timeofday', 'timeofmonth', 'timeofyear', 'dayofweek'], 'datetime_mappings': {}}, 'fit': {}}

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "include": self.include,
                "datetime_mappings": self.datetime_mappings,
            },
        )

        return json_dict

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = DatetimeInfoExtractor(
        ...     columns=["a", "b"],
        ...     include=["timeofday", "timeofmonth"],
        ... )

        >>> transformer.get_feature_names_out()
        ['a_timeofday', 'a_timeofmonth', 'b_timeofday', 'b_timeofmonth']

        ```

        """
        return [
            col + "_" + include_option
            for col in self.columns
            for include_option in self.include
        ]

    def _check_provided_mappings(
        self,
        datetime_mappings: Optional[dict[DatetimeInfoOptionStr, dict[int, str]]],
    ) -> None:
        """Process user provided mappings.

        Sets datetime_mappings attribute, then validates against RANGE_TO_MAP.

        Raises
        ------
            ValueError: keys in datetime mapping do not match values in include

        Examples
        --------
        ```pycon
        >>> transformer = DatetimeInfoExtractor(
        ...     columns="a",
        ...     include="timeofday",
        ... )

        >>> transformer._check_provided_mappings(
        ...     {
        ...         "timeofday": {
        ...             **{i: "start" for i in range(0, 12)},
        ...             **{i: "end" for i in range(12, 24)},
        ...         }
        ...     }
        ... )

        ```

        """
        if datetime_mappings:
            for key in datetime_mappings:
                if key not in self.include:
                    msg = f"{self.classname()}: keys in datetime_mappings should be in include"
                    raise ValueError(msg)

                # check provided mappings fit required format
                if set(datetime_mappings[key].keys()) != self.RANGE_TO_MAP[key]:
                    msg = f"{self.classname()}: {key} mapping dictionary should contain mapping for all values between {min(self.RANGE_TO_MAP[key])}-{max(self.RANGE_TO_MAP[key])}. {self.RANGE_TO_MAP[key] - set(datetime_mappings[key].keys())} are missing"
                    raise ValueError(msg)

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform - Extracts new features from datetime variables.

        Parameters
        ----------
        X : DataFrame
            Data with columns to extract info from.

        Returns
        -------
        X : DataFrame
            Transformed input X with added columns of extracted information.

        Example:
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = DatetimeInfoExtractor(
        ...     columns="a",
        ...     include="timeofmonth",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.datetime(1993, 9, 27), datetime.datetime(2005, 10, 7)],
        ...         "b": [datetime.datetime(1991, 5, 22), datetime.datetime(2001, 12, 10)],
        ...     },
        ... )

        >>> transformer.transform(test_df)
        shape: (2, 3)
        ┌─────────────────────┬─────────────────────┬───────────────┐
        │ a                   ┆ b                   ┆ a_timeofmonth │
        │ ---                 ┆ ---                 ┆ ---           │
        │ datetime[μs]        ┆ datetime[μs]        ┆ enum          │
        ╞═════════════════════╪═════════════════════╪═══════════════╡
        │ 1993-09-27 00:00:00 ┆ 1991-05-22 00:00:00 ┆ end           │
        │ 2005-10-07 00:00:00 ┆ 2001-12-10 00:00:00 ┆ start         │
        └─────────────────────┴─────────────────────┴───────────────┘

        ```

        """
        X = super().transform(X, return_native_override=False)

        # initialise mappings attr with defaults,
        # and overwrite with user provided mappings
        # where possible
        final_datetime_mappings = copy.deepcopy(self.DEFAULT_MAPPINGS)
        for key in self.datetime_mappings:
            final_datetime_mappings[key] = copy.deepcopy(
                self.datetime_mappings[key],
            )

        # this is a situation where we know the values our mappings allow,
        # so enum type is more appropriate than categorical and we
        # will cast to this at the end
        enums = {
            include_option: nw.Enum(
                sorted(set(final_datetime_mappings[include_option].values())),
            )
            for include_option in self.include
        }

        mappings_dict = {
            col + "_" + include_option: final_datetime_mappings[include_option]
            for col in self.columns
            for include_option in self.include
        }

        transform_dict = {
            col + "_" + include_option: (
                getattr(
                    nw.col(col).dt,
                    self.DATETIME_ATTR[include_option],
                )().replace_strict(
                    mappings_dict[col + "_" + include_option],
                )
            )
            for col in self.columns
            for include_option in self.include
        }

        # final casts
        transform_dict = {
            col + "_" + include_option: transform_dict[col + "_" + include_option].cast(
                enums[include_option],
            )
            for col in self.columns
            for include_option in self.include
        }

        X = X.with_columns(
            **transform_dict,
        )

        # Drop original columns if self.drop_original is True
        X = DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class DatetimeComponentOptions(str, Enum):
    """Contains options for DatetimeComponentExtractor."""

    __slots__ = ()

    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


DatetimeComponentOptionStr = Annotated[
    str,
    Is[lambda s: s in DatetimeComponentOptions._value2member_map_],
]
DatetimeComponentOptionList = Annotated[
    list,
    Is[
        lambda list_value: all(
            entry in DatetimeComponentOptions._value2member_map_ for entry in list_value
        )
    ],
]


class DatetimeComponentExtractor(BaseDatetimeTransformer):
    """Transformer to extract numeric datetime components.

    Attributes:
    ----------
    columns: List[str]
        List of columns for processing

    include : list of str
        Which numeric datetime components to extract

    polars_compatible : bool
        Indicates whether transformer has been converted to polars/pandas agnostic framework

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    jsonable: bool
        Indicates if transformer supports to/from_json methods

    FITS: bool
        Indicates whether transform requires fit to be run first

    Example:
    -------
    ```pycon
    >>> transformer = DatetimeComponentExtractor(
    ...     columns="a",
    ...     include=["hour", "day"],
    ... )
    >>> transformer
    DatetimeComponentExtractor(columns=['a'], include=['hour', 'day'])

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = transformer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'DatetimeComponentExtractor', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'dummy', 'drop_original': False, 'include': ['hour', 'day']}, 'fit': {}}

    >>> DatetimeComponentExtractor.from_json(json_dump)
    DatetimeComponentExtractor(columns=['a'], include=['hour', 'day'])

    ```

    """

    INCLUDE_OPTIONS: ClassVar[list[str]] = ["hour", "day", "month", "year"]

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        include: Union[DatetimeComponentOptionList, DatetimeComponentOptionStr],
        **kwargs: Union[str, bool],
    ) -> None:
        """Initialize the DatetimeComponentExtractor.

        Parameters
        ----------
        columns : str or list
            datetime columns to extract information from

        include : list of str
            Which numeric datetime components to extract

        new_column_name : str, default = "dummy"
            Name given to new column created by the transformation.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        if isinstance(include, str):
            include = [include]

        if "new_column_name" in kwargs:
            warnings.warn(
                f"{self.classname()}: new_column_name arg is unused by this transformer",
                stacklevel=2,
            )
            kwargs.pop("new_column_name", None)

        super().__init__(
            columns=columns,
            new_column_name="dummy",
            **kwargs,
        )

        self.include = include

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            List of features modified/created by the transformer


        Examples
        --------
        ```pycon
        >>> transformer = DatetimeComponentExtractor(
        ...     columns=["a", "b"],
        ...     include=["hour", "day"],
        ... )

        >>> transformer.get_feature_names_out()
        ['a_hour', 'a_day', 'b_hour', 'b_day']

        ```

        """
        return [
            col + "_" + include_option
            for col in self.columns
            for include_option in self.include
        ]

    def to_json(self) -> dict[str, Any]:
        """Convert transformer to JSON format.

        Returns
        -------
        dict:
            JSON representation of the transformer

        Examples
        --------
        ```pycon
        >>> transformer = DatetimeComponentExtractor(
        ...     columns="a",
        ...     include=["hour", "day"],
        ... )

        >>> transformer.to_json()
        {'tubular_version': '...', 'classname': 'DatetimeComponentExtractor', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'dummy', 'drop_original': False, 'include': ['hour', 'day']}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()
        json_dict["init"]["include"] = self.include
        return json_dict

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform - Extracts numeric datetime components.

        Parameters
        ----------
        X : DataFrame
            Data with columns to extract info from.

        Returns
        -------
        X : DataFrame
            Transformed input X with added columns of extracted information.


        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = DatetimeComponentExtractor(
        ...     columns="a",
        ...     include=["hour", "day"],
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [
        ...             datetime.datetime(1993, 9, 27, 14, 30),
        ...             datetime.datetime(2005, 10, 7, 9, 45),
        ...         ],
        ...         "b": [
        ...             datetime.datetime(1991, 5, 22, 18, 0),
        ...             datetime.datetime(2001, 12, 10, 23, 59),
        ...         ],
        ...     },
        ... )

        >>> transformer.transform(test_df)
        shape: (2, 4)
        ┌─────────────────────┬─────────────────────┬────────┬───────┐
        │ a                   ┆ b                   ┆ a_hour ┆ a_day │
        │ ---                 ┆ ---                 ┆ ---    ┆ ---   │
        │ datetime[μs]        ┆ datetime[μs]        ┆ f32    ┆ f32   │
        ╞═════════════════════╪═════════════════════╪════════╪═══════╡
        │ 1993-09-27 14:30:00 ┆ 1991-05-22 18:00:00 ┆ 14.0   ┆ 27.0  │
        │ 2005-10-07 09:45:00 ┆ 2001-12-10 23:59:00 ┆ 9.0    ┆ 7.0   │
        └─────────────────────┴─────────────────────┴────────┴───────┘

        ```

        """
        X = super().transform(X, return_native_override=False)

        transform_dict = {
            col + "_" + include_option: (
                getattr(
                    nw.col(col).dt,
                    include_option,
                )().cast(nw.Float32)  # can't cast to int as may have nulls
            )
            for col in self.columns
            for include_option in self.include
        }

        X = X.with_columns(
            **transform_dict,
        )

        return _return_narwhals_or_native_dataframe(X, self.return_native)


class DatetimeSinusoidUnitsOptions(str, Enum):
    """Options for units argument of DatetimeSinusoidCalculator."""

    __slots__ = ()

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    MICROSECOND = "microsecond"


DatetimeSinusoidUnitsOptionStr = Annotated[
    str,
    Is[lambda s: s in DatetimeSinusoidUnitsOptions._value2member_map_],
]


class MethodOptions(str, Enum):
    """Options for method arg of DatetimeSinusoidCalculator."""

    __slots__ = ()

    SIN = "sin"
    COS = "cos"


MethodOptionStr = Annotated[
    str,
    Is[lambda s: s in MethodOptions._value2member_map_],
]

MethodOptionList = Annotated[
    list,
    Is[
        lambda list_value: all(
            entry in MethodOptions._value2member_map_ for entry in list_value
        )
    ],
]

NumberNotBool = Annotated[
    Union[int, float],
    Is[
        # exclude bools which would pass isinstance(..., (float, int))
        lambda value: type(value) in {int, float}
    ],
]


@register
class DatetimeSinusoidCalculator(BaseDatetimeTransformer):
    """Calculate the sine or cosine of a datetime column in a given unit (e.g hour).

    Includes the option to scale period of the sine or cosine to match the natural
    period of the unit (e.g. 24).

    Attributes:
    ----------
    columns : str or list
        Columns to take the sine or cosine of.

    method : str or list
        The function to be calculated; either sin, cos or a list containing both.

    units : str or dict
        Which time unit the calculation is to be carried out on. Will take any of 'year', 'month',
        'day', 'hour', 'minute', 'second', 'microsecond'. Can be a string or a dict containing key-value pairs of column
        name and units to be used for that column.

    period : str, float or dict, default = 2*np.pi
        The period of the output in the units specified above. Can be a string or a dict containing key-value pairs of column
        name and units to be used for that column.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Example:
    -------
    ```pycon
    >>> DatetimeSinusoidCalculator(
    ...     columns="a",
    ...     method="sin",
    ...     units="month",
    ... )
    DatetimeSinusoidCalculator(columns=['a'], method=['sin'], units='month')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: Union[str, list[str]],
        method: Union[MethodOptionStr, MethodOptionList],
        units: Union[
            DatetimeSinusoidUnitsOptionStr,
            dict[str, DatetimeSinusoidUnitsOptionStr],
        ],
        period: Union[NumberNotBool, dict[str, NumberNotBool]] = 2 * np.pi,
        drop_original: bool = False,
        **kwargs: Union[bool, str],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list
            Columns to take the sine or cosine of. Must be a datetime[64] column.

        method : str or list
            Argument to specify which function is to be calculated. Accepted values are 'sin', 'cos' or a list containing both.

        units : str or dict
            Which time unit the calculation is to be carried out on. Accepted values are 'year', 'month',
            'day', 'hour', 'minute', 'second', 'microsecond'.  Can be a string or a dict containing key-value pairs of column
            name and units to be used for that column.

        period : int, float or dict, default = 2*np.pi
            The period of the output in the units specified above. To leave the period of the sinusoid output as 2 pi, specify 2*np.pi (or leave as default).
            Can be a string or a dict containing key-value pairs of column name and period to be used for that column.

        drop_original: bool
            indicates whether to drop original columns

        kwargs: Union[bool, str]
            arguments for base classes, e.g. verbose

        Raises
        ------
            ValueError: if keys in provided period dictionary do match provided columns

        """
        if "new_column_name" in kwargs:
            warnings.warn(
                f"{self.classname()}: new_column_name arg is unused by this transformer",
                stacklevel=2,
            )
            kwargs.pop("new_column_name", None)

        super().__init__(
            columns=columns,
            drop_original=drop_original,
            new_column_name="dummy",
            **kwargs,
        )

        method_list = [method] if isinstance(method, str) else method

        self.method = method_list
        self.units = units
        self.period = period

        if isinstance(units, dict) and sorted(units.keys()) != sorted(self.columns):
            msg = f"{self.classname()}: unit dictionary keys must be the same as columns but got {set(units.keys())}"
            raise ValueError(msg)

        if isinstance(period, dict) and sorted(period.keys()) != sorted(self.columns):
            msg = f"{self.classname()}: period dictionary keys must be the same as columns but got {set(period.keys())}"
            raise ValueError(msg)

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = DatetimeSinusoidCalculator(
        ...     columns="a",
        ...     method="sin",
        ...     units="month",
        ... )

        >>> transformer.get_feature_names_out()
        ['sin_6.283185307179586_month_a']

        ```

        """
        return [
            f"{method}_{self.period if not isinstance(self.period, dict) else self.period[column]}_{self.units if not isinstance(self.units, dict) else self.units[column]}_{column}"
            for column in self.columns
            for method in self.method
        ]

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Examples
        --------
        ```pycon
        >>> transformer = DatetimeSinusoidCalculator(
        ...     columns="a",
        ...     method="sin",
        ...     units="month",
        ... )
        >>> transformer.to_json()
        {'tubular_version': ..., 'classname': 'DatetimeSinusoidCalculator', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'dummy', 'drop_original': False, 'method': ['sin'], 'units': 'month', 'period': 6.283185307179586}, 'fit': {}}

        ```

        """
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "method": self.method,
                "units": self.units,
                "period": self.period,
            }
        )

        return json_dict

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: Optional[bool] = None,
    ) -> DataFrame:
        """Transform - creates column containing sine or cosine of another datetime column.

        Which function is used is stored in the self.method attribute.

        Parameters
        ----------
        X : pd/pl/nw.DataFrame
            Data to transform.

        return_native_override: Optional[bool]
            Option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : pd/pl/nw.DataFrame
            Input X with additional columns added, these are named "<method>_<original_column>"

        Example:
        --------
        ```pycon
        >>> import polars as pl
        >>> import datetime

        >>> transformer = DatetimeSinusoidCalculator(
        ...     columns="a",
        ...     method="sin",
        ...     units="month",
        ... )

        >>> test_df = pl.DataFrame(
        ...     {
        ...         "a": [datetime.datetime(1993, 9, 27), datetime.datetime(2005, 10, 7)],
        ...         "b": [datetime.datetime(1991, 5, 22), datetime.datetime(2001, 12, 10)],
        ...     },
        ... )

        >>> transformer.transform(test_df)
        shape: (2, 3)
        ┌─────────────────────┬─────────────────────┬───────────────────────────────┐
        │ a                   ┆ b                   ┆ sin_6.283185307179586_month_a │
        │ ---                 ┆ ---                 ┆ ---                           │
        │ datetime[μs]        ┆ datetime[μs]        ┆ f64                           │
        ╞═════════════════════╪═════════════════════╪═══════════════════════════════╡
        │ 1993-09-27 00:00:00 ┆ 1991-05-22 00:00:00 ┆ 0.412118                      │
        │ 2005-10-07 00:00:00 ┆ 2001-12-10 00:00:00 ┆ -0.544021                     │
        └─────────────────────┴─────────────────────┴───────────────────────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        return_native = self._process_return_native(return_native_override)

        X = super().transform(X, return_native_override=False)

        exprs = {}
        for column in self.columns:
            if not isinstance(self.units, dict):
                desired_units = self.units
            elif isinstance(self.units, dict):
                desired_units = self.units[column]
            if not isinstance(self.period, dict):
                desired_period = self.period
            elif isinstance(self.period, dict):
                desired_period = self.period[column]

            for method in self.method:
                new_column_name = f"{method}_{desired_period}_{desired_units}_{column}"

                # Calculate the sine or cosine of the column in the desired unit
                expr = getattr(
                    nw.col(column).dt,
                    desired_units,
                )() * (2 * np.pi / desired_period)

                expr = (
                    expr.map_batches(
                        lambda s: np.sin(
                            s.to_numpy(),
                        ),
                        return_dtype=nw.Float64,
                    )
                    if method == "sin"
                    else expr.map_batches(
                        lambda s: np.cos(
                            s.to_numpy(),
                        ),
                        return_dtype=nw.Float64,
                    )
                )
                expr = expr.alias(new_column_name)
                exprs[new_column_name] = expr

        X = X.with_columns(**exprs)
        # Drop original columns if self.drop_original is True
        X = DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
            return_native=False,
        )
        return _return_narwhals_or_native_dataframe(X, return_native)


# DEPRECATED TRANSFORMERS


@deprecated(
    "This Transformer is deprecated, use DateDifferenceTransformer instead. "
    "If you prefer this transformer to DateDifferenceTransformer, "
    "let us know through a github issue",
)
class DateDiffLeapYearTransformer(BaseGenericDateTransformer):
    """Transformer to calculate the number of years between two dates.

    !!! warning "Deprecated"
        This transformer is now deprecated; use `DateDifferenceTransformer` instead.

    Attributes
    ----------
    columns : List[str]
        List of 2 columns. First column will be subtracted from second.

    new_column_name : str, default = None
        Name given to calculated datediff column. If None then {column_upper}_{column_lower}_datediff
        will be used.

    drop_original : bool
        Indicator whether to drop old columns during transform method.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    deprecated: bool
        indicates if class has been deprecated

    """

    polars_compatible = True

    lazyframe_compatible = False

    FITS = False

    jsonable = False

    deprecated = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        new_column_name: str,
        missing_replacement: Optional[Union[float, int, str]] = None,
        drop_original: bool = False,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : List[str]
            List of 2 columns. First column will be subtracted from second.

        new_column_name : str
            Name for the new year column.

        drop_original : bool
            Flag for whether to drop the original columns.

        missing_replacement : int/float/str
            Value to output if either the lower date value or the upper date value are
            missing. Default value is None.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(
            columns=columns,
            new_column_name=new_column_name,
            drop_original=drop_original,
            **kwargs,
        )

        self.missing_replacement = missing_replacement

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = columns[0]
        self.column_upper = columns[1]

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Calculate year gap between the two provided columns.

        New column is created under the 'new_column_name', and optionally removes the
        old date columns.

        Parameters
        ----------
        X : pd/pl/nw.DataFrame
            Data containing self.columns

        Returns
        -------
        X : pd/pl/nw.DataFrame
            Data containing self.columns

        """
        X = nw.from_native(super().transform(X))

        # Create a helping column col0 for the first date. This will convert the date into an integer in a format or YYYYMMDD
        X = X.with_columns(
            (
                nw.col(self.columns[0]).cast(nw.Date).dt.year().cast(nw.Int64) * 10000
                + nw.col(self.columns[0]).cast(nw.Date).dt.month().cast(nw.Int64) * 100
                + nw.col(self.columns[0]).cast(nw.Date).dt.day().cast(nw.Int64)
            ).alias("col0"),
        )
        # Create a helping column col1 for the second date. This will convert the date into an integer in a format or YYYYMMDD
        X = X.with_columns(
            (
                nw.col(self.columns[1]).cast(nw.Date).dt.year().cast(nw.Int64) * 10000
                + nw.col(self.columns[1]).cast(nw.Date).dt.month().cast(nw.Int64) * 100
                + nw.col(self.columns[1]).cast(nw.Date).dt.day().cast(nw.Int64)
            ).alias("col1"),
        )

        # Compute difference between integers and if the difference is negative then adjust.
        # Finally divide by 10000 to get the years.
        X = X.with_columns(
            nw.when(nw.col("col1") < nw.col("col0"))
            .then(((nw.col("col0") - nw.col("col1")) // 10000) * (-1))
            .otherwise((nw.col("col1") - nw.col("col0")) // 10000)
            .cast(nw.Int64)
            .alias(self.new_column_name),
        ).drop(["col0", "col1"])

        # When we get a missing then replace with missing_replacement otherwise return the above calculation
        if self.missing_replacement is not None:
            X = X.with_columns(
                nw.when(
                    (nw.col(self.columns[0]).is_null())
                    | (nw.col(self.columns[1]).is_null()),
                )
                .then(
                    self.missing_replacement,
                )
                .otherwise(
                    nw.col(self.new_column_name),
                )
                .cast(nw.Int64)
                .alias(self.new_column_name),
            )

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
        )


@deprecated(
    """This transformer has not been selected for conversion to polars/narwhals,
    and so has been deprecated. If aspects of it have been useful to you, please raise an issue
    for it to be replaced with more specific transformers
    """,
)
class SeriesDtMethodTransformer(BaseDatetimeTransformer):
    """Transformer that applies a pandas.Series.dt method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.dt method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Attributes
    ----------
    column : str
        Name of column to apply transformer to. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : str
        Column name for transformation.

    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    pd_method_kwargs : dict
        Dictionary of keyword arguments to call the pd.Series.dt method with.

    drop_original: bool
        Indicates whether to drop self.column post transform

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

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
        new_column_name: str,
        pd_method_name: str,
        columns: Union[
            ListOfOneStr,
            str,
        ],
        pd_method_kwargs: Optional[GenericKwargs] = None,
        drop_original: bool = False,
        **kwargs: Optional[bool],
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        new_column_name : str
            The name of the column to be assigned to the output of running the pandas method in transform.

        pd_method_name : str
            The name of the pandas.Series.dt method to call.

        columns : str
            Column to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. This is avoid likely
            when the user forget to set columns, in this case all columns would be picked up when super
            transform runs.

        pd_method_kwargs : dict, default = {}
            A dictionary of keyword arguments to be passed to the pd.Series.dt method when it is called.

        drop_original: bool
            Indicates whether to drop self.column post transform

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.__init__().

        Raises
        ------
            AttributeError: if requested pd.Series.dt method does not exist

        """
        super().__init__(
            columns=columns,
            new_column_name=new_column_name,
            drop_original=drop_original,
            **kwargs,
        )

        if pd_method_kwargs is None:
            pd_method_kwargs = {}

        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs

        try:
            ser = pd.Series(
                [datetime.datetime(2020, 12, 21, tzinfo=datetime.timezone.utc)],
            )
            getattr(ser.dt, pd_method_name)

        except Exception as err:
            msg = f'{self.classname()}: error accessing "dt.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.dt method'
            raise AttributeError(msg) from err

        if callable(getattr(ser.dt, pd_method_name)):
            self._callable = True

        else:
            self._callable = False

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = self.columns[0]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specific column on input pandas.DataFrame (X) using the given pandas.Series.dt method.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pd.Series.dt method
        when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. These contain the output of
            running the pd.Series.dt method.

        """
        X = super().transform(X)

        if self._callable:
            X[self.new_column_name] = getattr(
                X[self.columns[0]].dt,
                self.pd_method_name,
            )(**self.pd_method_kwargs)

        else:
            X[self.new_column_name] = getattr(
                X[self.columns[0]].dt,
                self.pd_method_name,
            )

        # Drop original columns if self.drop_original is True
        return DropOriginalMixin.drop_original_column(
            X,
            self.drop_original,
            self.columns,
        )
