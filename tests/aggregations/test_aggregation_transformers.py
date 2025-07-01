import pytest

from tests import utils as u
from tubular.aggregations import AggregateRowOverColumnsTransformer


class BaseTransformerTest:
    """Base test class for transformer initializations."""

    def setup_transformer(self, columns, aggregations, key, drop_original=False):
        """Utility method to initialize a transformer."""
        return AggregateRowOverColumnsTransformer(
            columns,
            aggregations,
            key,
            drop_original,
        )


class TestAggregateRowOverColumnsTransformerInit(BaseTransformerTest):
    """Tests for AggregateRowOverColumnsTransformer initialization."""

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_invalid_key_error(self, library):
        """Test that an error is raised if the key column is not found."""
        columns = ["col1", "col2"]
        aggregations = ["min", "max"]
        key = "missing_key"

        # Create a DataFrame using the library parameter
        df_dict = {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        }
        df = u.dataframe_init_dispatch(df_dict, library)

        transformer = self.setup_transformer(columns, aggregations, key)
        with pytest.raises(
            ValueError,
            match=f"key '{key}' not found in dataframe columns",
        ):
            transformer.transform(df)


class TestAggregateRowOverColumnsTransformerMethodsTransform(BaseTransformerTest):
    """Tests for methods in AggregateRowOverColumnsTransformer."""

    @pytest.mark.parametrize("library", ["pandas", "polars"])
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
                    "col1_mean": [2.0, 3.0, 2.0, 3.0],
                    "col1_median": [2.0, 3.0, 2.0, 3.0],
                    "col1_sum": [4, 6, 4, 6],
                    "col1_count": [2, 2, 2, 2],
                    "col2_min": [5, 6, 5, 6],
                    "col2_max": [7, 8, 7, 8],
                    "col2_mean": [6.0, 7.0, 6.0, 7.0],
                    "col2_median": [6.0, 7.0, 6.0, 7.0],
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
                    "col1_mean": [2.0, 3.0, 2.0, 3.0],
                    "col1_median": [2.0, 3.0, 2.0, 3.0],
                    "col1_sum": [4, 6, 4, 6],
                    "col1_count": [2, 2, 2, 2],
                    "col2_min": [5, 6, 5, 6],
                    "col2_max": [7, 8, 7, 8],
                    "col2_mean": [6.0, 7.0, 6.0, 7.0],
                    "col2_median": [6.0, 7.0, 6.0, 7.0],
                    "col2_sum": [12, 14, 12, 14],
                    "col2_count": [2, 2, 2, 2],
                },
            ),
        ],
    )
    def test_transform(self, library, aggregations, drop_original, expected_data):
        """Test transform method aggregates rows correctly."""
        columns = ["col1", "col2"]
        key = "group_key"

        # Create a pandas DataFrame first
        df_dict = {
            "col1": [1, 2, 3, 4],
            "col2": [5, 6, 7, 8],
            "group_key": ["A", "B", "A", "B"],
        }

        df = u.dataframe_init_dispatch(df_dict, library)

        transformer = self.setup_transformer(columns, aggregations, key, drop_original)
        transformed_df = transformer.transform(df)

        # Create expected DataFrame using the library parameter
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Compare the transformed DataFrame with the expected DataFrame using the dispatch function
        u.assert_frame_equal_dispatch(transformed_df, expected_df)
