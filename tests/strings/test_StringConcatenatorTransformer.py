import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    SeparatorInitMixintests,
)
from tubular.strings import StringConcatenator


class TestInit(
    SeparatorInitMixintests,
    ColumnStrListInitTests,
    NewColumnNameInitMixintests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringConcatenator"


class TestTransform(GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringConcatenator"

    @staticmethod
    def test_correct_df_returned_1():
        """Test that correct df is returned after transformation."""
        df = d.create_df_1()
        expected_df = df.copy()

        x = StringConcatenator(
            columns=["a", "b"],
            new_column_name="merged_values",
        )

        df_transformed = x.transform(df)
        expected_df["merged_values"] = ["1 a", "2 b", "3 c", "4 d", "5 e", "6 f"]

        ta.equality.assert_frame_equal_msg(
            df_transformed,
            expected_df,
            "Incorrect dataframe returned after StringConcatenator transform",
        )

    @staticmethod
    def test_correct_df_returned_2():
        """Test that correct df is returned after transformation."""
        df = d.create_df_1()
        expected_df = df.copy()

        x = StringConcatenator(
            columns=["a", "b"],
            new_column_name="merged_values",
            separator=":",
        )

        df_transformed = x.transform(df)
        expected_df["merged_values"] = ["1:a", "2:b", "3:c", "4:d", "5:e", "6:f"]

        ta.equality.assert_frame_equal_msg(
            df_transformed,
            expected_df,
            "Incorrect dataframe returned after StringConcatenator transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringConcatenator"
