import pytest
import test_aide as ta

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
    def test_expected_output(self, test_dataframe):
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
    def test_expected_output_dropped(self, test_dataframe):
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
