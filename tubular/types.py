"""Generic typehints used throughout package."""

from enum import Enum
from typing import Annotated, Union

import narwhals as nw
import pandas as pd
import polars as pl
from beartype.vale import Is

DataFrame = Union[
    pd.DataFrame,
    pl.DataFrame,
    pl.LazyFrame,
    nw.DataFrame,
    nw.LazyFrame,
]

NarwhalsFrame = Union[nw.DataFrame, nw.LazyFrame]

Series = Union[
    pd.Series,
    pl.Series,
    nw.Series,
]

NumericTypes = [
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.Float64,
    nw.Float32,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.UInt128,
]

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

PositiveNumber = Annotated[
    Union[int, float],
    Is[lambda v: v > 0],
]

PositiveInt = Annotated[int, Is[lambda i: i >= 0]]

FloatBetweenZeroOne = Annotated[float, Is[lambda i: (i > 0) & (i < 1)]]

GenericKwargs = Annotated[
    dict[str, Union[int, float, str, list[int], list[str], list[float]]],
    Is[lambda dict_arg: all(isinstance(key, str) for key in dict_arg)],
]


class TimeUnitsOptions(str, Enum):
    """Enumeration of time unit options."""

    DAYS = "D"
    HOURS = "h"
    MINUTES = "m"
    SECONDS = "s"


TimeUnitsOptionsStr = Annotated[
    str,
    Is[lambda s: s in TimeUnitsOptions._value2member_map_],
]

ListOfTwoStrs = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 2],
]


class FloatTypeOptions(Enum):
    """Options for float dtypes."""

    Float32 = "Float32"
    Float64 = "Float64"


FloatTypeAnnotated = Annotated[
    str,
    Is[lambda dtype: dtype in FloatTypeOptions._value2member_map_],
]
