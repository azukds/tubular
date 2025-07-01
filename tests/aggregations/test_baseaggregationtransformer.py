import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import ColumnStrListInitTests, DropOriginalInitMixinTests
from tubular.aggregations import BaseAggregationTransformer


class TestBaseAggregationTransformerInit(
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Tests for BaseAggregationTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseAggregationTransformer"

    def test_invalid_aggregation_error(self):
        """Test that an error is raised for invalid aggregation methods."""
        columns = ["col1", "col2"]
        invalid_aggregations = ["invalid", "max"]
        with pytest.raises(BeartypeCallHintParamViolation):
            BaseAggregationTransformer(columns, invalid_aggregations)

    def test_columns_empty_list_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = []
        with pytest.raises(ValueError):
            uninitialized_transformers[self.transformer_name](**args)

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


class TestBaseAggregationTransformerCreateNewColNames:
    """Tests for methods in BaseAggregationTransformer."""

    def test_create_new_col_names(self):
        """Test create_new_col_names method returns correct column names."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max", "mean"]
        transformer = BaseAggregationTransformer(columns, aggregations)
        prefix = "agg"
        expected_col_names = ["agg_min", "agg_max", "agg_mean"]
        assert transformer.create_new_col_names(prefix) == expected_col_names
