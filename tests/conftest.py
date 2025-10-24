from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tubular.base as base
from tests.test_data import (
    create_aggregate_over_rows_test_df,
    create_is_between_dates_df_1,
    create_numeric_df_1,
    create_numeric_df_2,
    create_object_df,
)

if TYPE_CHECKING:
    import pandas as pd

"""
How To Use This Testing Framework
-----------------------------------

Fixtures in this file are used in tests for shared behaviour which can be run on multiple transformers through inheritance.
setup_class is defined in each test class to define cls.transformer_name as the name of the transformer being tested.

Test methods for shared behaviour are defined in base_tests.py or child test classes which are themselves inherited.  The
fixtures in this file can be used in these tests to access initiated/ uninitiated transformers and minimum attributes for
intiatialization.

Test classes which can import tests for shared behaviour inherit from an appropriate parent and define cls.transformer_name
in setup_class so that these fixtures will retrieve the appropriate transformer instance. setup_class is called by pytest
at a class level before its test methods are run https://docs.pytest.org/en/8.0.x/how-to/xunit_setup.html.

New transformers will need to be added to minimal_attribute_dict.
 """


def get_all_classes(
    wanted_module: str | None = None,
) -> dict[str, base.BaseTransformer]:
    """Method to call the weighted_quantile method and prepare the outputs.

    If there are no None values in the supplied quantiles then the outputs from weighted_quantile
    are returned as is. If there are then prepare_quantiles removes the None values before
    calling weighted_quantile and adds them back into the output, in the same position, after
    calling.

    Parameters
    ----------
    wanted_module : str or None
        str indicating which module to load classes from (e.g. 'tubular.dates'). If none, loads from all modules.

    Returns
    -------
    all_classes : dict[str, BaseTransformer]
        Dictionary containing classes in format {transformer_name:transformer}

    """
    root = str(Path(__file__).parent.parent)

    all_classes = []
    modules_to_ignore = [
        "tests",
        "conftest",
        "setup",
    ]

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=[root],
    ):
        if wanted_module and modname != wanted_module:
            continue
        mod_parts = modname.split(".")
        if any(part in modules_to_ignore for part in mod_parts) or "_" in modname:
            continue
        module = import_module(modname)
        classes = inspect.getmembers(module, inspect.isclass)

        classes = [
            (name, transformer)
            for name, transformer in classes
            if issubclass(transformer, base.BaseTransformer)
        ]

        all_classes.extend(classes)

    return set(all_classes)


@pytest.fixture()
def minimal_attribute_dict():
    """defines minmal attributes (values) needed to initiate each transformer named (key).
    New transformers need to be added here"""
    return {
        "ArbitraryImputer": {
            "columns": ["b"],
            "impute_value": 1,
        },
        "BaseCappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "BaseCrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "BaseCrossColumnNumericTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "BaseGenericDateTransformer": {
            "columns": ["a"],
            "new_column_name": "bla",
        },
        "BaseDatetimeTransformer": {
            "columns": ["a"],
            "new_column_name": "bla",
        },
        "BaseImputer": {
            "columns": ["b"],
        },
        "BaseMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            # this arg is dependent on inputs, so
            # may not generalise well
            "return_dtypes": {"a": "Int8"},
        },
        "BaseMappingTransformMixin": {
            "columns": ["a"],
        },
        "BaseNominalTransformer": {
            "columns": ["b"],
        },
        "BaseNumericTransformer": {
            "columns": ["a", "b"],
        },
        "BaseTransformer": {
            "columns": ["a"],
        },
        "BetweenDatesTransformer": {
            "new_column_name": "e",
            "columns": ["a", "c", "b"],
        },
        "CappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "ColumnDtypeSetter": {
            "columns": ["a"],
            "dtype": str,
        },
        "CrossColumnAddTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnMultiplyTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CutTransformer": {
            "new_column_name": "b",
            "column": "a",
            "cut_kwargs": {
                "bins": 3,
            },
        },
        "DataFrameMethodTransformer": {
            "columns": ["a", "b"],
            "new_column_names": "f",
            "pd_method_name": "sum",
            "drop_original": True,
        },
        "DateDifferenceTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "new_column",
        },
        "DateDiffLeapYearTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "new_column",
        },
        "DatetimeInfoExtractor": {
            "columns": ["a"],
        },
        "DatetimeSinusoidCalculator": {
            "columns": ["a"],
            "method": ["sin"],
            "units": "month",
        },
        "EqualityChecker": {
            "columns": ["a", "b"],
            "new_column_name": "c",
            "drop_original": True,
        },
        "GroupRareLevelsTransformer": {
            "columns": ["b"],
        },
        "InteractionTransformer": {
            "columns": ["a", "b"],
        },
        "LogTransformer": {
            "columns": ["a"],
        },
        "MappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            # this arg is dependent on inputs, so
            # may not generalise well
            "return_dtypes": {"a": "Int8"},
        },
        "MeanImputer": {
            "columns": ["b"],
        },
        "MeanResponseTransformer": {
            "columns": ["b"],
        },
        "MedianImputer": {
            "columns": ["b"],
        },
        "ModeImputer": {
            "columns": ["b"],
        },
        "NearestMeanResponseImputer": {
            "columns": ["b"],
        },
        "NominalToIntegerTransformer": {
            "columns": ["b"],
        },
        "NullIndicator": {
            "columns": ["a"],
        },
        "OneDKmeansTransformer": {
            "columns": "a",
            "new_column_name": "new",
        },
        "OneHotEncodingTransformer": {
            "columns": ["a", "b"],
            "drop_original": True,
            "separator": "-",
        },
        "OrdinalEncoderTransformer": {
            "columns": ["b"],
        },
        "OutOfRangeNullTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "PCATransformer": {
            "columns": ["a", "c"],
        },
        "ScalingTransformer": {
            "scaler_type": "standard",
            "columns": ["a", "b"],
        },
        "SeriesDtMethodTransformer": {
            "new_column_name": "new_column",
            "pd_method_name": "month",
            "columns": "b",
        },
        "SeriesStrMethodTransformer": {
            "columns": ["b"],
            "new_column_name": "a",
            "pd_method_name": "split",
        },
        "SetValueTransformer": {
            "value": 1,
            "columns": ["a"],
        },
        "StringConcatenator": {
            "columns": ["a", "b"],
            "new_column_name": "c",
            "separator": "-",
        },
        "ToDatetimeTransformer": {
            "columns": "a",
            "time_format": "%d/%m/%Y",
        },
        "TwoColumnOperatorTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
            "pd_method_name": "add",
        },
        "BaseAggregationTransformer": {
            "columns": ["a", "b"],
            "aggregations": ["min", "max"],
            "drop_original": False,
        },
        "AggregateRowsOverColumnTransformer": {
            "columns": ["a", "b"],
            "aggregations": ["min", "max"],
            "key": "c",
            "drop_original": False,
        },
        "AggregateColumnsOverRowTransformer": {
            "columns": ["a", "b"],
            "aggregations": ["min", "max"],
            "drop_original": False,
        },
        "DifferenceTransformer": {
            "columns": ["a", "b"],
        },
        "RatioTransformer": {
            "columns": ["a", "b"],
            "return_dtype": "Float32",  # Default dtype for the resulting column
        },
    }


@pytest.fixture()
def minimal_dataframe_lookup(request) -> dict[str, pd.DataFrame]:
    """links transformers to minimal dataframes needed to successfully run transformer. There is logic to do this automatically by module, so function will only need to be edited where either:
    - a new module that operates primarily on non-numeric columns is added
    - a new transformer is added to an existing module that breaks the pattern of that module, e.g. a transformer in dates.py that operates on numeric columns

    Returns
    -------

    min_df_dict: dict[str, pd.DataFrame]
        dictionary mapping transformers to minimal dataframes that they can successfully run on

    """

    # setup to default to pandas if not provided
    library = getattr(request, "param", "pandas")

    num_df = create_numeric_df_1(library=library)
    nan_df = create_numeric_df_2(library=library)
    object_df = create_object_df(library=library)
    date_df = create_is_between_dates_df_1(library=library)
    agg_df = create_aggregate_over_rows_test_df(
        library=library,
    )

    # generally most transformers will work with num_df
    min_df_dict = {x[0]: num_df for x in get_all_classes()}

    # override dict value with date_df for transformers in tubular.dates
    date_transformers = [x[0] for x in get_all_classes(wanted_module="tubular.dates")]
    for transformer in date_transformers:
        min_df_dict[transformer] = date_df

    # override dict value for transformers that run on object type columns
    object_modules = ["tubular.mapping", "tubular.nominal", "tubular.strings"]
    object_transformers = []
    for module in object_modules:
        object_transformers = object_transformers + [
            x[0] for x in get_all_classes(wanted_module=module)
        ]

    for transformer in object_transformers:
        min_df_dict[transformer] = object_df

    # Some may require further manual overwrites
    other_num_transformers = [
        "CrossColumnMultiplyTransformer",
        "CrossColumnAddTransformer",
        "BaseCrossColumnNumericTransformer",
        # prevent below two from being imported from import in nominal file
        "MedianImputer",
        "MeanImputer",
    ]
    for transformer in other_num_transformers:
        min_df_dict[transformer] = num_df

    # Some transformers require missing values to work
    other_nan_transformers = [
        "NearestMeanResponseImputer",
    ]
    for transformer in other_nan_transformers:
        min_df_dict[transformer] = nan_df

    min_df_dict["BaseAggregationTransformer"] = agg_df
    min_df_dict["AggregateRowsOverColumnTransformer"] = agg_df

    return min_df_dict


@pytest.fixture()
def initialized_transformers(minimal_attribute_dict):
    """dictionary of {transformer name : initiated transformer} pairs for all transformers"""
    return {x[0]: x[1](**minimal_attribute_dict[x[0]]) for x in get_all_classes()}


@pytest.fixture()
def uninitialized_transformers():
    """dictionary of {transformer name : uninitiated transformer} pairs for all transformers"""
    return {x[0]: x[1] for x in get_all_classes()}
