from enum import Enum
from typing import Annotated, Optional, Union

import numpy as np
from beartype.vale import Is

# needed as by default beartype will just randomly sample to type check elements
# and we want consistency
ListOfStrs = Annotated[
    list,
    Is[lambda list_arg: all(isinstance(l_value, str) for l_value in list_arg)],
]

NonEmptyListOfStrs = Annotated[list[str], Is[lambda list_arg: len(list_arg) > 0]]

ListOfOneStr = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 1],
]

ListOfTwoStrs = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 2],
]

ListOfThreeStrs = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 3],
]

Number = Union[int, float]

FloatBetween0And1 = Annotated[float, Is[lambda value: value >= 0 and value <= 1]]

PositiveNumber = Annotated[
    Union[int, float],
    Is[lambda v: v > 0],
]

PositiveInt = Annotated[int, Is[lambda i: i >= 0]]

GenericKwargs = Annotated[
    dict[str, Union[int, float, str, list[int], list[str], list[float]]],
    Is[lambda dict_arg: all(isinstance(key, str) for key in dict_arg)],
]


class TimeUnitsOptions(str, Enum):
    DAYS = "D"
    HOURS = "h"
    MINUTES = "m"
    SECONDS = "s"


TimeUnitsOptionsStr = Annotated[
    str,
    Is[lambda s: s in TimeUnitsOptions._value2member_map_],
]

CapValues = Annotated[
    list[Optional[Number]],
    Is[
        lambda cap_values: (
            # two capping vals
            (len(cap_values) == 2)
            &
            # not all None
            (any(element is not None for element in cap_values))
            &
            # arg 1<arg 2 if both not None
            (
                any(element is None for element in cap_values)
                or cap_values[0] < cap_values[1]
            )
            &
            # not nan or inf
            all(
                not (np.isnan(value) or np.isinf(value))
                for value in cap_values
                if value is not None
            )
            &
            # check all are numeric if not None
            all(
                isinstance(value, (float, int))
                for value in cap_values
                if value is not None
            )
        )
    ],
]

CapValuesDict = Annotated[
    dict[str, CapValues],
    Is[lambda cap_values_dict: (all(isinstance(key, str) for key in cap_values_dict))],
]

QuantileCaps = Annotated[
    list[Optional[Number]],
    Is[
        lambda quantile_caps: (
            # two capping vals
            (len(quantile_caps) == 2)
            &
            # not all None
            (any(element is not None for element in quantile_caps))
            &
            # arg 1<arg 2 if both not None
            (
                any(element is None for element in quantile_caps)
                or quantile_caps[0] < quantile_caps[1]
            )
            &
            # not nan or inf
            all(
                not (np.isnan(value) or np.isinf(value))
                for value in quantile_caps
                if value is not None
            )
            &
            # check all are numeric if not None
            all(
                isinstance(value, (float, int))
                for value in quantile_caps
                if value is not None
            )
            & all(
                (value >= 0) and (value <= 1)
                for value in quantile_caps
                if value is not None
            )
        )
    ],
]

# TODO - put these into a function for nice error handlign
# and apply all at dict level for consistent handling
QuantileCapsDict = Annotated[
    dict[str, QuantileCaps],
    Is[
        lambda cap_values_dict: (
            (all(isinstance(key, str) for key in cap_values_dict))
            & all(
                (value >= 0) and (value <= 1)
                for quantile_caps in cap_values_dict.values()
                for value in quantile_caps
                if value is not None
            )
        )
    ],
]
