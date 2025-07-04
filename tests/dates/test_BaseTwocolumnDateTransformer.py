import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tests.dates.test_BaseGenericDateTransformer import GenericDatesMixinTransformTests


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    TwoColumnListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"

    # overload until we beartype the new_column_name mixin
    @pytest.mark.parametrize(
        "new_column_type",
        [1, True, {"a": 1}, [1, 2], np.inf, np.nan],
    )
    def test_new_column_name_type_error(
        self,
        new_column_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if any type other than str passed to new_column_name"""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["new_column_name"] = new_column_type

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestTransform(GenericTransformTests, GenericDatesMixinTransformTests):
    """Tests for BaseTwoColumnDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"
