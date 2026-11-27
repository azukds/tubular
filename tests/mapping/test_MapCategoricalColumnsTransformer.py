from tests.base_tests import (
    EmptyMappingsFitTransformPassTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)
from tests.mapping.test_BaseMappingTransformer import (
    BaseMappingTransformerInitTests,
    BaseMappingTransformerTransformTests,
)


class TestInit(BaseMappingTransformerInitTests):
    """Tests for transformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MapCategoricalColumnsTransformer"


class TestFit(GenericFitTests):
    """Tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MapCategoricalColumnsTransformer"


class TestTransform(BaseMappingTransformerTransformTests):
    """Tests for transformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MapCategoricalColumnsTransformer"


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyMappingsFitTransformPassTests,
):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MapCategoricalColumnsTransformer"
