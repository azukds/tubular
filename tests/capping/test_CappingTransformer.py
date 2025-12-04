import pytest

from tests.base_tests import OtherBaseBehaviourTests
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)


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
