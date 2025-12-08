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

    @staticmethod
    def get_CappingTransformer() -> CappingTransformer:
        return CappingTransformer(capping_values={"AveOccup": [1, 5.5]})

    @staticmethod
    def get_OutOfRangeNullTransformer() -> OutOfRangeNullTransformer:
        return OutOfRangeNullTransformer(capping_values={"AveOccup": [1, 5.5]})

    @staticmethod
    def get_ArbitraryImputer() -> ArbitraryImputer:
        return ArbitraryImputer(
            columns=["HouseAge_1", "AveOccup_1", "Population_1"],
            impute_value=-1,
        )

    @staticmethod
    def get_MedianImputer() -> MedianImputer:
        return MedianImputer(columns=["HouseAge_2", "AveOccup_2", "Population_2"])

    @staticmethod
    def get_MeanImputer() -> MeanImputer:
        return MeanImputer(columns=["HouseAge_3", "AveOccup_3", "Population_3"])

    @staticmethod
    def get_ModeImputer() -> ModeImputer:
        return ModeImputer(columns=["HouseAge_4", "AveOccup_4", "Population_4"])

    @staticmethod
    def get_NearestMeanResponseImputer() -> NearestMeanResponseImputer:
        return NearestMeanResponseImputer(
            columns=["HouseAge_5", "AveOccup_5", "Population_5"],
        )

    @staticmethod
    def get_NullIndicator() -> NullIndicator:
        return NullIndicator(columns=["HouseAge_6", "AveOccup_6", "Population_6"])

    @staticmethod
    def get_MappingTransformer() -> MappingTransformer:
        return MappingTransformer(mappings={"categorical_1": {"a": "c", "b": "d"}})

    @staticmethod
    def get_NominalToIntegerTransformer() -> NominalToIntegerTransformer:
        return NominalToIntegerTransformer(columns=["categorical_2"])

    @staticmethod
    def get_MeanResponseTransformer() -> MeanResponseTransformer:
        return MeanResponseTransformer(columns=["categorical_3"])

    @staticmethod
    def get_GroupRareLevelsTransformer() -> GroupRareLevelsTransformer:
        return GroupRareLevelsTransformer(columns=["categorical_4"])

    @staticmethod
    def get_OneHotEncodingTransformer() -> OneHotEncodingTransformer:
        return OneHotEncodingTransformer(columns=["categorical_ohe"])

    @staticmethod
    def get_LogTransformer() -> LogTransformer:
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
