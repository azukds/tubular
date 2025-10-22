import numpy as np
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tubular.comparison import EqualityChecker


class TestInit(
    DropOriginalInitMixinTests,
    NewColumnNameInitMixintests,
    TwoColumnListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "EqualityChecker"

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
        cls.transformer_name = "EqualityChecker"


class TestTransform(GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "EqualityChecker"

    @pytest.mark.parametrize(
        "test_dataframe",
        [d.create_df_5(), d.create_df_2(), d.create_df_9()],
    )
    @staticmethod
    def test_expected_output(test_dataframe):
        """Tests that the output given by EqualityChecker tranformer is as you would expect
        when all cases are neither all True nor False.
        """
        expected = test_dataframe
        expected["bool_logic"] = expected["b"] == expected["c"]

        example_transformer = EqualityChecker(
            columns=["b", "c"],
            new_column_name="bool_logic",
        )
        actual = example_transformer.transform(test_dataframe)

        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="EqualityChecker transformer does not produce the expected output",
            print_actual_and_expected=True,
        )

    @pytest.mark.parametrize(
        "test_dataframe",
        [d.create_df_5(), d.create_df_2(), d.create_df_9()],
    )
    @staticmethod
    def test_expected_output_dropped(test_dataframe):
        """Tests that the output given by EqualityChecker tranformer is as you would expect
        when all cases are neither all True nor False.
        """
        expected = test_dataframe.copy()
        expected["bool_logic"] = expected["b"] == expected["c"]
        expected = expected.drop(["b", "c"], axis=1)

        example_transformer = EqualityChecker(
            columns=["b", "c"],
            new_column_name="bool_logic",
            drop_original=True,
        )
        actual = example_transformer.transform(test_dataframe)

        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="EqualityChecker transformer does not produce the expected output",
            print_actual_and_expected=True,
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "EqualityChecker"
