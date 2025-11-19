import numpy as np
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

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

    # overload this test until we have converted separator mixin
    # to beartype
    @pytest.mark.parametrize(
        "separator",
        [1, True, {"a": 1}, [1, 2], None, np.inf, np.nan],
    )
    def test_separator_type_error(
        self,
        separator,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if any type other than str passed to separator"""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["separator"] = separator

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestTransform(GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringConcatenator"

    def test_correct_df_returned_1(self):
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

    def test_correct_df_returned_2(self):
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
