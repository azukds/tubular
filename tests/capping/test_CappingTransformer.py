import copy
from typing import ClassVar

import numpy as np
import pytest

from tests.base_tests import OtherBaseBehaviourTests
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)
from tests.utils import dataframe_init_dispatch


class TestInit(GenericCappingInitTests):
    """Tests for CappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestFit(GenericCappingFitTests):
    """Tests for CappingTransformer.fit()."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_benchmark_many_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        benchmark,
    ):
        """benchmark performance for many row transforms"""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args.pop("capping_values")

        args["quantiles"] = {"a": [0.01, 0.99], "b": [0.8, 0.85]}

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)

        df_dict = {
            "a": rng1.integers(0, 100, size=100),
            "b": rng2.integers(0, 1000, size=100),
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        _ = benchmark(transformer.fit, df)

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestTransform(GenericCappingTransformTests):
    """Tests for CappingTransformer.transform()."""

    benchmark_capping: ClassVar = {
        "capping_values": {"a": [2, 20], "b": [40, 99]},
        "quantiles": {"a": [0.01, 0.99], "b": [0.8, 0.85]},
    }

    @pytest.mark.benchmark
    @pytest.mark.parametrize("capping", ["capping_values", "quantiles"])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_benchmark_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        capping,
        benchmark,
    ):
        """benchmark performance for single row transforms"""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])

        if capping != "capping_values":
            args.pop("capping_values")

        args[capping] = self.benchmark_capping[capping]

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [20],
            "b": [2],
        }
        single_row_df = dataframe_init_dispatch(single_row_df_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        transformer.fit(single_row_df)

        _ = benchmark(transformer.transform, single_row_df)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("capping", ["capping_values", "quantiles"])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_benchmark_many_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        capping,
        benchmark,
    ):
        """benchmark performance for many row transforms"""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        if capping != "capping_values":
            args.pop("capping_values")

        args[capping] = self.benchmark_capping[capping]

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)

        df_dict = {
            "a": rng1.integers(0, 100, size=100),
            "b": rng2.integers(0, 1000, size=100),
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        transformer.fit(df)

        _ = benchmark(transformer.transform, df)

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    def test_get_params_call_with_capping_values_none(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test get_params method when capping_values is None."""
        args = minimal_attribute_dict[self.transformer_name]
        args["capping_values"] = None
        args["quantiles"] = {"a": [0.1, 0.9]}
        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Ensure no AttributeError is raised when calling get_params method
        try:
            transformer.get_params()
        except AttributeError as e:
            pytest.fail(f"AttributeError was raised: {e}")
