import numpy as np
import pandas as pd

import tests.test_data as d
from tests.base_tests import OtherBaseBehaviourTests
from tests.mapping.test_BaseCrossColumnNumericTransformer import (
    BaseCrossColumnNumericTransformerInitTests,
    BaseCrossColumnNumericTransformerTransformTests,
)
from tests.utils import assert_frame_equal_dispatch


class TestInit(BaseCrossColumnNumericTransformerInitTests):
    """Tests for CrossColumnAddTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnAddTransformer"


class TestTransform(BaseCrossColumnNumericTransformerTransformTests):
    """Tests for CrossColumnAddTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnAddTransformer"

    def expected_df_1(self):
        """Expected output from test_expected_output."""
        return pd.DataFrame(
            {"a": [2.1, 3.2, 4.3, 5.4, 6.5, 7.6], "b": ["a", "b", "c", "d", "e", "f"]},
        )

    def expected_df_3(self):
        """Expected output from test_multiple_mappings_expected_output."""
        df = pd.DataFrame(
            {
                "a": [4.1, 5.1, 4.1, 4, 8, 10.2, 7, 8, 9, np.nan],
                "b": ["a", "a", "a", "d", "e", "f", "g", np.nan, np.nan, np.nan],
                "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", np.nan],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def test_expected_output(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that transform is giving the expected output."""
        df = d.create_df_1()
        expected = self.expected_df_1()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "a"
        args["copy"] = True

        x = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

        for i in range(len(df)):
            row = df.iloc[[i]]
            row_transformed = x.transform(row)
            row_expected = expected.iloc[[i]]

            assert_frame_equal_dispatch(row_transformed, row_expected)

    def test_multiple_mappings_expected_output(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that mappings by multiple columns are both applied in transform."""
        df = d.create_df_5()
        expected = self.expected_df_3()

        mapping = {"b": {"a": 1.1, "f": 1.2}, "c": {"a": 2, "e": 3}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "a"
        args["copy"] = True

        x = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

        for i in range(len(df)):
            row = df.iloc[[i]]
            row_transformed = x.transform(row)
            row_expected = expected.iloc[[i]]

            assert_frame_equal_dispatch(row_transformed, row_expected)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CrossColumnAddTransformer"
