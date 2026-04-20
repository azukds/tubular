import numpy as np
import pytest

import tests.test_data as d
from tests import utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    EmptyColumnsFitTransformPassTests,
    FailedFitWeightFilterTest,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    OtherBaseBehaviourTestsNumeric,
    ReturnNativeTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
)
from tubular.imputers import MedianImputer


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests, FailedFitWeightFilterTest):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values(self, library, lazy):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3(library=library)

        transformer = MedianImputer(columns=["a", "b", "c"])

        transformer.fit(u._convert_to_lazy(df, lazy))

        assert transformer.impute_values_ == {
            "a": df["a"].median(),
            "b": df["b"].median(),
            "c": df["c"].median(),
        }, "impute_values_ attribute"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weighted(self, library, lazy):
        """Test that the impute values learnt during fit are expected - when using weights."""
        df = d.create_df_9_with_null_weight_row(library=library)

        transformer = MedianImputer(columns=["a"], weights_column="c")

        transformer.fit(u._convert_to_lazy(df, lazy))

        assert transformer.impute_values_ == {
            "a": np.int64(4),
        }, "impute_values_ attribute"


class TestTransform(
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
    GenericTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyColumnsFitTransformPassTests,
    OtherBaseBehaviourTestsNumeric,
):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"
