import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import OtherBaseBehaviourTests
from tests.mapping.test_BaseCrossColumnNumericTransformer import (
    BaseCrossColumnNumericTransformerInitTests,
    BaseCrossColumnNumericTransformerTransformTests,
)


class TestInit(BaseCrossColumnNumericTransformerInitTests):
    """Tests for CrossColumnMultiplyTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnMultiplyTransformer"


class TestTransform(BaseCrossColumnNumericTransformerTransformTests):
    """Tests for CrossColumnMultiplyTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnMultiplyTransformer"

    def expected_df_1():
        """Expected output for test_expected_output."""
        return pd.DataFrame(
            {"a": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6], "b": ["a", "b", "c", "d", "e", "f"]},
        )

    def expected_df_2():
        """Expected output for test_non_specified_values_unchanged."""
        return pd.DataFrame(
            {"a": [1.1, 2.4, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]},
        )

    def expected_df_3():
        """Expected output for test_multiple_mappings_ordered_dict."""
        df = pd.DataFrame(
            {
                "a": [2.2, 4.4, 3.3, 4, 15, 21.6, 7, 8, 9, np.nan],
                "b": ["a", "a", "a", "d", "e", "f", "g", np.nan, np.nan, np.nan],
                "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", np.nan],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_1()),
    )
    def test_expected_output(
        self,
        df,
        expected,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that transform is giving the expected output."""
        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "a"

        x = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from cross column multiply transformer",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_5(), expected_df_3()),
    )
    def test_multiple_mappings_expected_output(
        self,
        df,
        expected,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that mappings by multiple columns are both applied in transform."""
        mapping = {"b": {"a": 1.1, "f": 1.2}, "c": {"a": 2, "e": 3}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "a"

        x = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from cross column multiply transformer",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnMultiplyTransformer"
