import polars as pl
import pytest

from tests.base_tests import EmptyCappingsFitTransformPassTests, OtherBaseBehaviourTests
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.capping import CappingTransformer


class TestInit(GenericCappingInitTests):
    """Tests for CappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestFit(GenericCappingFitTests):
    """Tests for CappingTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestTransform(GenericCappingTransformTests):
    """Tests for CappingTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestLazyYSupport:
    """Tests for lazy y support in CappingTransformer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that CappingTransformer accepts LazyFrame for y parameter."""
        df_dict = {"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
        df = dataframe_init_dispatch(df_dict, library)

        y_lazy = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

        transformer = CappingTransformer(quantiles={"b": [0.1, 0.9]})

        # Fit should accept lazy y and not raise an error
        transformer.fit(df, y_lazy)

        # Transform should apply caps correctly based on learned quantiles
        expected = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.0, 2.0, 3.0, 4.0, 4.5],
            }
        )

        transformed = transformer.transform(df)

        assert_frame_equal_dispatch(transformed, expected)


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests, EmptyCappingsFitTransformPassTests
):
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
