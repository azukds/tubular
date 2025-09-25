import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericTransformTests,
)
from tests.utils import dataframe_init_dispatch


class TestBaseAggregationTransformerInit(
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Tests for BaseAggregationTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseAggregationTransformer"

    def test_invalid_aggregation_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an error is raised for invalid aggregation methods."""

        args = minimal_attribute_dict[self.transformer_name]
        args["aggregations"] = ["invalid", "max"]
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)

    # NOTE - can delete this test once DropOriginalMixin is converted to beartype
    @pytest.mark.parametrize("drop_orginal_column", (0, "a", ["a"], {"a": 10}, None))
    def test_drop_column_arg_errors(
        self,
        drop_orginal_column,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["drop_original"] = drop_orginal_column
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):  # Adjust to expect BeartypeCallHintParamViolation
            uninitialized_transformers[self.transformer_name](**args)


class TestBaseAggregationTransformerTransform(GenericTransformTests):
    "tests for BaseAggregationTransformer.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseAggregationTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "bad_values",
        [
            [True, False, None],
            ["a", "b", "c"],
        ],
    )
    def test_error_for_bad_col_types(
        self,
        library,
        bad_values,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        "test that errors are thrown at transform for non numeric columns"

        df_dict = {
            "a": bad_values,
        }

        test_df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        args = minimal_attribute_dict[self.transformer_name]
        args["columns"] = ["a"]
        transformer = uninitialized_transformers[self.transformer_name](**args)

        msg = r"attempting to call transformer on non-numeric columns \['a'\], which is not supported"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(test_df)
