import pytest

import tests.test_data as d
from tests.base_tests import OtherBaseBehaviourTests
from tests.mapping.test_BaseCrossColumnMappingTransformer import (
    BaseCrossColumnMappingTransformerInitTests,
    BaseCrossColumnMappingTransformerTransformTests,
)


class BaseCrossColumnNumericTransformerInitTests(
    BaseCrossColumnMappingTransformerInitTests,
):
    """
    Tests for BaseCrossColumnNumericTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_mapping_values_not_numeric_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if mappings values are not numeric."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["adjust_column"] = "c"

        args["mappings"] = {
            "b": {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f"},
        }

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: mapping values must be numeric",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    def test_inferred_return_dtypes(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """test that return_dtypes are inferred correctly if not provided - test is
        overloaded as these transformers can only handle numeric types
        """

        kwargs = minimal_attribute_dict[self.transformer_name]
        kwargs["mappings"] = {
            "a": {"a": 1, "b": 2},
            "c": {"d": 1.0, "e": 2.0},
        }
        kwargs["return_dtypes"] = None

        transformer = uninitialized_transformers[self.transformer_name](
            **kwargs,
        )

        expected = {
            "a": "Int64",
            "c": "Float64",
        }

        actual = transformer.return_dtypes

        assert actual == expected, (
            f"return_dtypes attr not inferred as expected, expected {expected} but got {actual}"
        )


class BaseCrossColumnNumericTransformerTransformTests(
    BaseCrossColumnMappingTransformerTransformTests,
):
    """
    Tests for the transform method on BaseCrossColumnnumericTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_adjust_col_not_numeric_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if the adjust_column is not numeric."""
        df = d.create_df_2()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["adjust_column"] = "c"

        x = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: variable c must have numeric dtype.",
        ):
            x.transform(df)


class TestInit(BaseCrossColumnNumericTransformerInitTests):
    """Tests for BaseCrossColumnNumeicTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCrossColumnNumericTransformer"


class TestTransform(BaseCrossColumnNumericTransformerTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCrossColumnNumericTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCrossColumnNumericTransformer"
