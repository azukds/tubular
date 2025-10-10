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
    "ArbitraryImputer",
    "BetweenDatesTransformer",
    "CappingTransformer",
    "DateDifferenceTransformer",
    "DatetimeInfoExtractor",
    "DatetimeSinusoidCalculator",
    "GroupRareLevelsTransformer",
    "MappingTransformer",
    "MeanImputer",
    "MeanResponseTransformer",
    "MedianImputer",
    "ModeImputer",
    "NullIndicator",
    "OneDKmeansTransformer",
    "OneHotEncodingTransformer",
    "OutOfRangeNullTransformer",
    "SetValueTransformer",
    "ToDatetimeTransformer",
]

with suppress(ModuleNotFoundError):
    __version__ = version("tubular")
