import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    EmptyColumnsFitTransformPassTests,
    FailedFitWeightFilterTest,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
)
from tests.utils import _convert_to_lazy, dataframe_init_dispatch
from tubular.imputers import MeanImputer


class TestInit(WeightColumnInitMixinTests, ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests, FailedFitWeightFilterTest):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values(self, library, lazy):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3(library=library)

        x = MeanImputer(columns=["a", "b", "c"])

        x.fit(_convert_to_lazy(df, lazy))

        expected_impute_values = {
            "a": df["a"].mean(),
            "b": df["b"].mean(),
            "c": df["c"].mean(),
        }

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_attr not as expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weighted(self, library, lazy):
        """Test that the impute values learnt during fit are expected - when weights are used."""
        df = d.create_df_9_with_null_weight_row(library=library)

        x = MeanImputer(columns=["a", "b"], weights_column="c")

        x.fit(_convert_to_lazy(df, lazy))

        expected_impute_values = {
            "a": (3 + 4 + 16 + 36) / (3 + 2 + 4 + 6),
            "b": (10 + 4 + 12 + 10 + 6) / (2 + 1 + 4 + 5 + 6),
        }

        assert x.impute_values_ == expected_impute_values, (
            f"learnt impute_values_ attr not as expected, expected {expected_impute_values} but got {x.impute_values_}"
        )


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTestsWeight,
    GenericImputerTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"


class TestLazyYSupport:
    """Tests for lazy y support in MeanImputer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that MeanImputer accepts LazyFrame for y parameter."""
        df_dict = {"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, None, 4.0, 5.0]}
        df = dataframe_init_dispatch(df_dict, library)

        y_lazy = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

        transformer = MeanImputer(columns="b")

        # Should not raise an error
        transformer.fit(df, y_lazy)


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests, EmptyColumnsFitTransformPassTests
):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"
