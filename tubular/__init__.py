from contextlib import suppress
from importlib.metadata import version

from tubular.aggregations import (
    AggregateColumnsOverRowTransformer,
    AggregateRowsOverColumnTransformer,
)
from tubular.capping import CappingTransformer, OutOfRangeNullTransformer
from tubular.dates import (
    BetweenDatesTransformer,
    DateDifferenceTransformer,
    DatetimeInfoExtractor,
    DatetimeSinusoidCalculator,
    ToDatetimeTransformer,
)
from tubular.imputers import (
    ArbitraryImputer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    NullIndicator,
)
from tubular.mapping import MappingTransformer
from tubular.misc import SetValueTransformer
from tubular.nominal import (
    GroupRareLevelsTransformer,
    MeanResponseTransformer,
    OneHotEncodingTransformer,
)
from tubular.numeric import OneDKmeansTransformer

__all__ = [
    "AggregateColumnsOverRowTransformer",
    "AggregateRowsOverColumnTransformer",
    "CappingTransformer",
    "OutOfRangeNullTransformer",
    "DateDifferenceTransformer",
    "ToDatetimeTransformer",
    "BetweenDatesTransformer",
    "DatetimeInfoExtractor",
    "DatetimeSinusoidCalculator",
    "ArbitraryImputer",
    "MeanImputer",
    "ModeImputer",
    "MedianImputer",
    "NullIndicator",
    "MappingTransformer",
    "SetValueTransformer",
    "GroupRareLevelsTransformer",
    "MeanResponseTransformer",
    "OneHotEncodingTransformer",
    "OneDKmeansTransformer",
]

with suppress(ModuleNotFoundError):
    __version__ = version("tubular")
