import copy

import narwhals as nw
import pytest

from tests import utils as u
from tests.aggregations.test_BaseAggregationTransformer import (
    TestBaseAggregationTransformerInit,
    TestBaseAggregationTransformerTransform,
)


class TestAggregateColumnsOverRowTransformerInit(TestBaseAggregationTransformerInit):
    """Tests for init method in AggregateColumnsOverRowTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateColumnsOverRowTransformer"


class TestAggregateColumnsOverRowTransformerTransform(
    TestBaseAggregationTransformerTransform,
):
    """Tests for transform method in AggregateColumnsOverRowTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateColumnsOverRowTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "aggregations, expected_data",
        [
            (
                ["min", "max", "mean", "sum"],
                {
                    "a": [1, 2, 3, 4, 8],
                    "b": [2, 3, 4, 5, 9],
                    "c": [3, 4, 5, 6, 10],
                    "a_b_c_min": [1, 2, 3, 4, 8],
                    "a_b_c_max": [3, 4, 5, 6, 10],
                    "a_b_c_mean": [2.0, 3.0, 4.0, 5.0, 9.0],
                    "a_b_c_sum": [6, 9, 12, 15, 27],
                },
            ),
        ],
    )
    def test_transform_basic_case_outputs(
        self,
        library,
        aggregations,
        expected_data,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method aggregates rows correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b", "c"]
        args["aggregations"] = aggregations

        df_dict = {
            "a": [1, 2, 3, 4, 8],
            "b": [2, 3, 4, 5, 9],
            "c": [3, 4, 5, 6, 10],
        }

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        # transformer = transformer_setup(columns, aggregations, key, drop_original)
        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Create expected DataFrame using the library parameter
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Compare the transformed DataFrame with the expected DataFrame using the dispatch function
        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max", "mean", "sum"]

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [None],
            "b": [2],
            "c": ["A"],
        }
        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)

        # cast none row to numeric type
        single_row_df = nw.from_native(single_row_df)
        single_row_df = single_row_df.with_columns(
            nw.col("a").cast(nw.Float64),
        ).to_native()

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(single_row_df)

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": [None],
            "b": [2],
            "c": ["A"],
            "a_b_min": [2.0],
            "a_b_max": [2.0],
            "a_b_mean": [2.0],
            "a_b_sum": [2.0],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        # ensure none columns are numeric type
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(nw.col(col).cast(nw.Float64) for col in ["a"])
            .to_native()
        )

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_with_nulls(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max", "mean", "sum"]

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [1.0, None, 3.0, None, 8.0],
            "b": [None, 3.0, None, 5.0, 9.0],
            "c": ["A", "B", "A", "B", "A"],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df_with_nulls)

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [1, None, 3, None, 8],
            "b": [None, 3, None, 5, 9],
            "c": ["A", "B", "A", "B", "A"],
            "a_b_min": [1.0, 3.0, 3.0, 5.0, 8.0],
            "a_b_max": [1.0, 3.0, 3.0, 5.0, 9.0],
            "a_b_mean": [1.0, 3.0, 3.0, 5.0, 8.5],
            "a_b_sum": [1.0, 3.0, 3.0, 5.0, 17.0],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
