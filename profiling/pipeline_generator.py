from typing import Optional

from sklearn.pipeline import Pipeline

from tubular.capping import CappingTransformer, OutOfRangeNullTransformer
from tubular.imputers import (
    ArbitraryImputer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    NearestMeanResponseImputer,
    NullIndicator,
)
from tubular.mapping import MappingTransformer
from tubular.nominal import (
    GroupRareLevelsTransformer,
    MeanResponseTransformer,
    NominalToIntegerTransformer,
    OneHotEncodingTransformer,
)
from tubular.numeric import LogTransformer


class TubularPipelineGenerator:
    """Class to generate pipelines containing any combination of the tubular transformers that are also found in fubular."""

    def __init__(self) -> None:
        self.all_tubular_transformers = [
            "CappingTransformer",
            "OutOfRangeNullTransformer",
            "ArbitraryImputer",
            "MedianImputer",
            "MeanImputer",
            "ModeImputer",
            "NearestMeanResponseImputer",
            "NullIndicator",
            "MappingTransformer",
            "NominalToIntegerTransformer",
            "MeanResponseTransformer",
            "GroupRareLevelsTransformer",
            "OneHotEncodingTransformer",
            "LogTransformer",
        ]

    def get_CappingTransformer(self) -> CappingTransformer:
        return CappingTransformer(capping_values={"AveOccup": [1, 5.5]})

    def get_OutOfRangeNullTransformer(self) -> OutOfRangeNullTransformer:
        return OutOfRangeNullTransformer(capping_values={"AveOccup": [1, 5.5]})

    def get_ArbitraryImputer(self) -> ArbitraryImputer:
        return ArbitraryImputer(
            columns=["HouseAge_1", "AveOccup_1", "Population_1"],
            impute_value=-1,
        )

    def get_MedianImputer(self) -> MedianImputer:
        return MedianImputer(columns=["HouseAge_2", "AveOccup_2", "Population_2"])

    def get_MeanImputer(self) -> MeanImputer:
        return MeanImputer(columns=["HouseAge_3", "AveOccup_3", "Population_3"])

    def get_ModeImputer(self) -> ModeImputer:
        return ModeImputer(columns=["HouseAge_4", "AveOccup_4", "Population_4"])

    def get_NearestMeanResponseImputer(self) -> NearestMeanResponseImputer:
        return NearestMeanResponseImputer(
            columns=["HouseAge_5", "AveOccup_5", "Population_5"],
        )

    def get_NullIndicator(self) -> NullIndicator:
        return NullIndicator(columns=["HouseAge_6", "AveOccup_6", "Population_6"])

    def get_MappingTransformer(self) -> MappingTransformer:
        return MappingTransformer(mappings={"categorical_1": {"a": "c", "b": "d"}})

    def get_NominalToIntegerTransformer(self) -> NominalToIntegerTransformer:
        return NominalToIntegerTransformer(columns=["categorical_2"])

    def get_MeanResponseTransformer(self) -> MeanResponseTransformer:
        return MeanResponseTransformer(columns=["categorical_3"])

    def get_GroupRareLevelsTransformer(self) -> GroupRareLevelsTransformer:
        return GroupRareLevelsTransformer(columns=["categorical_4"])

    def get_OneHotEncodingTransformer(self) -> OneHotEncodingTransformer:
        return OneHotEncodingTransformer(columns=["categorical_ohe"])

    def get_LogTransformer(self) -> LogTransformer:
        return LogTransformer(columns=["HouseAge_7", "AveOccup_7", "Population_7"])

    def generate_pipeline(
        self,
        transformers_to_include: Optional[list] = None,
        verbose: bool = False,
    ) -> Pipeline:
        if not transformers_to_include:
            transformers_to_include = self.all_tubular_transformers

        steps = [
            (transformer, getattr(self, f"get_{transformer}")())
            for transformer in transformers_to_include
        ]

        return Pipeline(steps, verbose=verbose)
