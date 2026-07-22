"""Stateless transforms for date columns."""

import copy
from enum import Enum
from typing import Annotated

import narwhals as nw
import numpy as np
from beartype import beartype
from beartype.vale import Is

from tubular.types import ListOfThreeStrs, ListOfTwoStrs

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


@beartype
def diff_two_dates(
    columns: ListOfTwoStrs,
    units: DateDifferenceUnitsOptionsStr,
    new_column_name: str,
    custom_days_divider: int | None = None,
) -> nw.Expr:
    """Get expression for calculating difference between provided dates.

    Parameters
    ----------
    columns:
        columns to difference, will calculate columns[1]-columns[0]
    units:
            Accepted values are "week", "fortnight", "lunar_month", "common_year", "custom_days", 'D', 'h', 'm', 's'
    custom_days_divider:
        Integer value for the "custom_days" unit
    new_column_name:
        name for output column

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    start_date_col = nw.col(columns[0])
    end_date_col = nw.col(columns[1])

    # truncating time for specific units
    if units in UNITS_TO_TRUNCATE_TIME_FOR:
        start_date_col = start_date_col.dt.truncate("1d")
        end_date_col = end_date_col.dt.truncate("1d")

    if units == "custom_days":
        timedelta_value, timedelta_format = custom_days_divider, "D"
        denominator = np.timedelta64(timedelta_value, timedelta_format)
    else:
        timedelta_value, timedelta_format = UNITS_TO_TIMEDELTA_PARAMS[units]
        denominator = np.timedelta64(timedelta_value, timedelta_format)

    return ((end_date_col - start_date_col) / denominator).alias(new_column_name)


@beartype
def convert_columns_to_datetime(
    columns: list[str], time_format: str | None = None
) -> list[nw.Expr]:
    """Get expression for calculating difference between provided dates.

    Parameters
    ----------
    columns:
        List of names of the column to convert to datetime.

    time_format:
        str indicating format of time to parse, e.g. '%d/%m/%Y'

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [nw.col(col).str.to_datetime(format=time_format) for col in columns]


@beartype
def check_if_three_date_columns_are_sequential(
    columns: ListOfThreeStrs,
    lower_inclusive: bool,
    upper_inclusive: bool,
    new_column_name: str,
) -> nw.Expr:
    """Get expression for checking if three date columns are sequential.

    Parameters
    ----------
    columns:
        List of names of the column to convert to datetime.

    lower_inclusive:
        If lower_inclusive is True the comparison to column_lower will be column_lower <=
        column_between, otherwise the comparison will be column_lower < column_between.

    upper_inclusive:
        If upper_inclusive is True the comparison to column_upper will be column_between <=
        column_upper, otherwise the comparison will be column_between < column_upper.

    new_column_name:
        name for output column.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    lower_comparison = (
        nw.col(columns[0]) <= nw.col(columns[1])
        if lower_inclusive
        else nw.col(columns[0]) < nw.col(columns[1])
    )

    upper_comparison = (
        nw.col(columns[1]) <= nw.col(columns[2])
        if upper_inclusive
        else nw.col(columns[1]) < nw.col(columns[2])
    )

    return (
        nw.when(nw.col(columns[0]) <= nw.col(columns[2]))
        .then(lower_comparison & upper_comparison)
        .otherwise(None)
        .alias(new_column_name)
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

DEFAULT_MAPPINGS = {
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

INCLUDE_OPTIONS = list(DEFAULT_MAPPINGS.keys())

RANGE_TO_MAP = {
    DatetimeInfoOptions.TIME_OF_DAY.value: set(range(24)),
    DatetimeInfoOptions.TIME_OF_MONTH.value: set(range(1, 32)),
    DatetimeInfoOptions.TIME_OF_YEAR.value: set(range(1, 13)),
    DatetimeInfoOptions.DAY_OF_WEEK.value: set(range(1, 8)),
}

DATETIME_ATTR = {
    DatetimeInfoOptions.TIME_OF_DAY.value: "hour",
    DatetimeInfoOptions.TIME_OF_MONTH.value: "day",
    DatetimeInfoOptions.TIME_OF_YEAR.value: "month",
    DatetimeInfoOptions.DAY_OF_WEEK.value: "weekday",
}


@beartype
def extract_datetime_info(
    columns: list[str],
    datetime_mappings: dict[DatetimeInfoOptionStr, dict[int, str]] | None = None,
    include: DatetimeInfoOptionList | DatetimeInfoOptionStr | None = None,
) -> list[nw.Expr]:
    """Get expression for extracting below components from datetime columns.

    - time of day
    - time of month
    - time of year
    - time of week

    Parameters
    ----------
    columns:
        List of names of the column to convert to datetime.


    include:
        Which datetime categorical information to extract

    datetime_mappings:
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
        then default values from DEFAULT_MAPPINGS will be used for this
        option.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    # initialise mappings with defaults,
    # and overwrite with user provided mappings
    # where possible
    final_datetime_mappings = copy.deepcopy(DEFAULT_MAPPINGS)
    for key in datetime_mappings:
        final_datetime_mappings[key] = copy.deepcopy(
            datetime_mappings[key],
        )

    # this is a situation where we know the values our mappings allow,
    # so enum type is more appropriate than categorical and we
    # will cast to this at the end
    enums = {
        include_option: nw.Enum(
            sorted(set(final_datetime_mappings[include_option].values())),
        )
        for include_option in include
    }

    mappings_dict = {
        col + "_" + include_option: final_datetime_mappings[include_option]
        for col in columns
        for include_option in include
    }

    return [
        (
            getattr(
                nw.col(col).dt,
                DATETIME_ATTR[include_option],
            )()
            .replace_strict(
                mappings_dict[col + "_" + include_option],
            )
            .cast(enums[include_option])
            .alias(col + "_" + include_option)
        )
        for col in columns
        for include_option in include
    ]


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


@beartype
def extract_datetime_components(
    columns: list[str],
    include: DatetimeComponentOptionList | DatetimeComponentOptionStr,
) -> list[nw.Expr]:
    """Get expression for extracting below components from datetime columns.

    - time of day
    - time of month
    - time of year
    - time of week

    Parameters
    ----------
    columns:
        List of names of the column to convert to datetime.

    include:
        Which datetime categorical information to extract

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    return [
        (
            getattr(
                nw.col(col).dt,
                include_option,
            )()
            .cast(
                nw.Float32  # can't cast to int as may have nulls
            )
            .alias(col + "_" + include_option)
        )
        for col in columns
        for include_option in include
    ]


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
    int | float,
    Is[
        # exclude bools which would pass isinstance(..., (float, int))
        lambda value: type(value) in {int, float}
    ],
]


@beartype
def extract_datetime_sinusoid_components(
    columns: list[str],
    period_dict: dict[str, NumberNotBool],
    units_dict: dict[str, DatetimeSinusoidUnitsOptionStr],
    method: MethodOptionStr | MethodOptionList,
) -> list[nw.Expr]:
    """Extract trig components from given units of datetime columns (e.g. month).

    Parameters
    ----------
    columns:
        Columns to take the sine or cosine of. Must be a datetime[64] column.

    method:
        Argument to specify which function is to be calculated. Accepted values are 'sin', 'cos' or a list containing both.

    units_dict:
        Which time unit the calculation is to be carried out on. Accepted values are 'year', 'month',
        'day', 'hour', 'minute', 'second', 'microsecond'. dict containing key-value pairs of column
        name and units to be used for that column.

    period_dict:
        The period of the output in the units specified above. To leave the period of the sinusoid output as 2 pi, specify 2*np.pi (or leave as default).
        dict containing key-value pairs of column name and period to be used for that column.

    Returns
    -------
    list[nw.Expr]: expressions for transformation

    """
    # first convert to desired units
    return [
        getattr(
            getattr(
                nw.col(column).dt,
                units_dict[column],
            )()
            * (2 * np.pi / period_dict[column]),
            trig_method,
        )().alias(f"{trig_method}_{period_dict[column]}_{units_dict[column]}_{column}")
        for column in columns
        for trig_method in method
    ]
