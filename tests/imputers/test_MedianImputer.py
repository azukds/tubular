import narwhals as nw
import numpy as np
import pytest

import tests.test_data as d
from tests import utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
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
from tubular.imputers import MedianImputer


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values(self, library):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3(library=library)

        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)

        # replace 'a' with all null values to trigger warning
        df = df.with_columns(
            nw.new_series(
                name="d",
                values=[None] * len(df),
                backend=native_backend,
            ),
        )

        df = df.to_native()

        transformer = MedianImputer(columns=["a", "b", "c", "d"])

        transformer.fit(df)

        assert transformer.impute_values_ == {
            "a": df["a"].median(),
            "b": df["b"].median(),
            "c": df["c"].median(),
            "d": None,
        }, "impute_values_ attribute"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weighted(self, library):
        """Test that the impute values learnt during fit are expected - when using weights."""
        df = d.create_df_9(library=library)

        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)

        # replace 'a' with all null values to trigger warning
        df = df.with_columns(
            nw.new_series(
                name="d",
                values=[None] * len(df),
                backend=native_backend,
            ),
        )

        df = df.to_native()

        transformer = MedianImputer(columns=["a", "d"], weights_column="c")

        transformer.fit(df)

        assert transformer.impute_values_ == {
            "a": np.int64(4),
            "d": None,
        }, "impute_values_ attribute"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_fit_not_changing_data(self, library):
        """Test fit does not change X."""
        df = d.create_df_1(library=library)

        transformer = MedianImputer(columns="a")

        transformer.fit(df)

        # Check whole dataframes
        u.assert_frame_equal_dispatch(
            d.create_df_1(library=library),
            df,
        )


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


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"
