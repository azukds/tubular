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
        columns = ["a", "b"]
        invalid_aggregations = ["invalid", "max"]
        with pytest.raises(BeartypeCallHintParamViolation):
            BaseAggregationTransformer(columns, invalid_aggregations)

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
        columns = ["a", "b"]
        aggregations = ["min", "max", "mean"]
        transformer = BaseAggregationTransformer(columns, aggregations)

        # Use the column name itself as the prefix
        expected_col_names = ["a_min", "a_max", "a_mean", "b_min", "b_max", "b_mean"]

        # Generate new column names using each column name as prefix
        new_col_names = []
        for column in columns:
            new_col_names.extend(transformer.create_new_col_names(column))

        assert new_col_names == expected_col_names
