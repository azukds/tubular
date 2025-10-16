import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.mapping.test_BaseMappingTransformer import (
    BaseMappingTransformerInitTests,
    BaseMappingTransformerTransformTests,
)


class BaseCrossColumnMappingTransformerInitTests(BaseMappingTransformerInitTests):
    """
    Tests for BaseCrossColumnMappingTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_adjust_columns_non_string_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if adjust_column is not a string."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["adjust_column"] = 1

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: adjust_column should be a string",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    def test_inferred_return_dtypes(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """test that return_dtypes are inferred correctly if not provided - test is
        overloaded as these transformers require OrderedDict for multicolumn mapping
        """

        kwargs = minimal_attribute_dict[self.transformer_name]
        kwargs["mappings"] = {
            "a": {"a": 1, "b": 2},
        }
        kwargs["return_dtypes"] = None

        transformer = uninitialized_transformers[self.transformer_name](
            **kwargs,
        )

        expected = {
            "a": "Int64",
        }

        actual = transformer.return_dtypes

        assert actual == expected, (
            f"return_dtypes attr not inferred as expected, expected {expected} but got {actual}"
        )


class BaseCrossColumnMappingTransformerTransformTests(
    BaseMappingTransformerTransformTests,
):
    """
    Tests for the transform method on BaseCrossColumnMappingTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def expected_df_2():
        """Expected output from test_non_specified_values_unchanged."""
        return pd.DataFrame(
            {"b": ["a", "b", "c", "d", "e", "f"]},
        )

    def test_adjust_col_not_in_x_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if the adjust_column is not present in the dataframe."""
        df = d.create_df_1()

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["adjust_column"] = "c"

        args["mappings"] = {
            "b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6},
        }

        x = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: variable c is not in X",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_2()),
    )
    def test_non_specified_values_unchanged(
        self,
        df,
        expected,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that values not specified in mappings are left unchanged in transform."""
        mapping = {"b": {"a": 1.1, "b": 1.2}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "a"

        x = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = x.transform(df)

        ta.equality.assert_series_equal_msg(
            actual=df_transformed["b"],
            expected=expected["b"],
            msg_tag=f"expected output from {self.transformer_name}",
        )


class TestInit(BaseCrossColumnMappingTransformerInitTests):
    """Tests for BaseCrossColumnMappingTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCrossColumnMappingTransformer"


class TestTransform(BaseCrossColumnMappingTransformerTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCrossColumnMappingTransformer"
