import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tubular.aggregations import BaseAggregationTransformer


class TestBaseAggregationTransformerInit:
    """Tests for BaseAggregationTransformer initialization."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max", "mean"]
        transformer = BaseAggregationTransformer(columns, aggregations)
        assert transformer.columns == columns
        assert transformer.aggregations == aggregations
        assert transformer.drop_original is False
        assert transformer.level == "row"

    def test_invalid_aggregation_error(self):
        """Test that an error is raised for invalid aggregation methods."""
        columns = ["col1", "col2"]
        invalid_aggregations = ["invalid", "max"]
        with pytest.raises(BeartypeCallHintParamViolation):
            BaseAggregationTransformer(columns, invalid_aggregations)

    def test_invalid_level_error(self):
        """Test that an error is raised for invalid level."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max"]
        with pytest.raises(BeartypeCallHintParamViolation):
            BaseAggregationTransformer(columns, aggregations, level="invalid")


class TestBaseAggregationTransformerMethods:
    """Tests for methods in BaseAggregationTransformer."""

    def test_create_new_col_names(self):
        """Test create_new_col_names method returns correct column names."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max", "mean"]
        transformer = BaseAggregationTransformer(columns, aggregations)
        prefix = "agg"
        expected_col_names = ["agg_min", "agg_max", "agg_mean"]
        assert transformer.create_new_col_names(prefix) == expected_col_names
