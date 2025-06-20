import narwhals as nw
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tubular.aggregations import AggregateRowOverColumnsTransformer


class TestAggregateRowOverColumnsTransformerInit:
    """Tests for AggregateRowOverColumnsTransformer initialization."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max", "mean"]
        key = "group_key"
        transformer = AggregateRowOverColumnsTransformer(columns, aggregations, key)
        assert transformer.columns == columns
        assert transformer.aggregations == aggregations
        assert transformer.key == key
        assert transformer.drop_original is False
        assert transformer.level == "row"

    def test_invalid_aggregation_type_error(self):
        """Test that an exception is raised for invalid aggregation type."""
        columns = ["col1", "col2"]
        aggregations = ["invalid_agg"]
        key = "group_key"

        with pytest.raises(BeartypeCallHintParamViolation):
            AggregateRowOverColumnsTransformer(columns, aggregations, key)

    def test_invalid_key_error(self):
        """Test that an error is raised if the key column is not found."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max"]
        key = "missing_key"

        # Create a pandas DataFrame first
        pd_df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
            },
        )

        # Convert to narwhals DataFrame using the conversion function
        df = nw.from_native(pd_df)

        # Assert the type to ensure it's a Narwhals DataFrame
        assert isinstance(df, nw.DataFrame), "DataFrame should be a Narwhals DataFrame"

        transformer = AggregateRowOverColumnsTransformer(columns, aggregations, key)
        with pytest.raises(
            ValueError,
            match=f"key '{key}' not found in dataframe columns",
        ):
            transformer.transform(df)


class TestAggregateRowOverColumnsTransformerMethods:
    """Tests for methods in AggregateRowOverColumnsTransformer."""

    @pytest.mark.parametrize(
        "aggregations, drop_original, expected_data",
        [
            # Test cases for "min", "max", "mean", "median", "sum", and "count"
            (
                ["min", "max", "mean", "median", "sum", "count"],
                False,
                {
                    "col1": [1, 2, 3, 4],
                    "col2": [5, 6, 7, 8],
                    "group_key": ["A", "B", "A", "B"],
                    "col1_min": [1, 2, 1, 2],
                    "col1_max": [3, 4, 3, 4],
                    "col1_mean": [2, 3, 2, 3],
                    "col1_median": [2, 3, 2, 3],
                    "col1_sum": [4, 6, 4, 6],
                    "col1_count": [2, 2, 2, 2],
                    "col2_min": [5, 6, 5, 6],
                    "col2_max": [7, 8, 7, 8],
                    "col2_mean": [6, 7, 6, 7],
                    "col2_median": [6, 7, 6, 7],
                    "col2_sum": [12, 14, 12, 14],
                    "col2_count": [2, 2, 2, 2],
                },
            ),
            (
                ["min", "max", "mean", "median", "sum", "count"],
                True,
                {
                    "group_key": ["A", "B", "A", "B"],
                    "col1_min": [1, 2, 1, 2],
                    "col1_max": [3, 4, 3, 4],
                    "col1_mean": [2, 3, 2, 3],
                    "col1_median": [2, 3, 2, 3],
                    "col1_sum": [4, 6, 4, 6],
                    "col1_count": [2, 2, 2, 2],
                    "col2_min": [5, 6, 5, 6],
                    "col2_max": [7, 8, 7, 8],
                    "col2_mean": [6, 7, 6, 7],
                    "col2_median": [6, 7, 6, 7],
                    "col2_sum": [12, 14, 12, 14],
                    "col2_count": [2, 2, 2, 2],
                },
            ),
        ],
    )
    def test_transform(self, aggregations, drop_original, expected_data):
        """Test transform method aggregates rows correctly."""
        columns = ["col1", "col2"]
        key = "group_key"

        # Create a pandas DataFrame first
        pd_df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": [5, 6, 7, 8],
                "group_key": ["A", "B", "A", "B"],
            },
        )

        df = nw.from_native(pd_df)

        transformer = AggregateRowOverColumnsTransformer(
            columns,
            aggregations,
            key,
            drop_original=drop_original,
        )
        transformed_df = transformer.transform(df)

        # Create expected DataFrame using pandas
        pd_expected_df = pd.DataFrame(expected_data)

        # Convert transformed DataFrame to pandas and ensure data types match
        transformed_pd_df = pd.DataFrame(
            transformed_df.values,
            columns=transformed_df.columns,
        )

        # Ensure data types match
        transformed_pd_df = transformed_pd_df.astype(pd_expected_df.dtypes.to_dict())

        # Assert DataFrame equality using pandas
        pd.testing.assert_frame_equal(
            transformed_pd_df.sort_index(axis=1),
            pd_expected_df.sort_index(axis=1),
        )
