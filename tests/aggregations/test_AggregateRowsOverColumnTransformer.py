import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.aggregations.test_BaseAggregationTransformer import (
    TestBaseAggregationTransformerInit,
    TestBaseAggregationTransformerTransform,
)
from tests.test_data import create_aggregate_over_rows_test_df


class TestAggregateRowsOverColumnTransformerInit(TestBaseAggregationTransformerInit):
    """Tests for init method in AggregateRowsOverColumnTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateRowsOverColumnTransformer"

    @pytest.mark.parametrize("key", (0, ["a"], {"a": 10}, None))
    def test_key_arg_errors(
        self,
        key,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["key"] = key
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):  # Adjust to expect BeartypeCallHintParamViolation
            uninitialized_transformers[self.transformer_name](**args)


class TestAggregateRowsOverColumnTransformerTransform(
    TestBaseAggregationTransformerTransform,
):
    """Tests for transform method in AggregateRowsOverColumnTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateRowsOverColumnTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_invalid_key_error(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an error is raised if the key column is not found."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max"]
        args["key"] = "missing_key"

        df = create_aggregate_over_rows_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        with pytest.raises(
            ValueError,
            match=f"key '{args['key']}' not found in dataframe columns",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "aggregations, expected_data",
        [
            # Test cases for "min", "max", "mean", "median", "sum", and "count"
            (
                ["min", "max", "mean", "median", "count", "sum"],
                {
                    "a": [1, 2, 3, 4, 8],
                    "b": [2, 3, 4, 5, 9],
                    "c": ["A", "B", "A", "B", "A"],
                    "a_min": [1, 2, 1, 2, 1],
                    "a_max": [8, 4, 8, 4, 8],
                    "a_mean": [4.0, 3.0, 4.0, 3.0, 4.0],
                    "a_median": [3.0, 3.0, 3.0, 3.0, 3.0],
                    "a_count": [3, 2, 3, 2, 3],
                    "a_sum": [12, 6, 12, 6, 12],
                    "b_min": [2, 3, 2, 3, 2],
                    "b_max": [9, 5, 9, 5, 9],
                    "b_mean": [5.0, 4.0, 5.0, 4.0, 5.0],
                    "b_median": [4.0, 4.0, 4.0, 4.0, 4.0],
                    "b_count": [3, 2, 3, 2, 3],
                    "b_sum": [15, 8, 15, 8, 15],
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
        args["columns"] = ["a", "b"]
        args["aggregations"] = aggregations
        args["key"] = "c"

        df = create_aggregate_over_rows_test_df(library=library)

        # transformer = transformer_setup(columns, aggregations, key, drop_original)
        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Create expected DataFrame using the library parameter
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                [
                    nw.col("a_count").cast(nw.UInt32),
                    nw.col("b_count").cast(nw.UInt32),
                ],
            ).to_native()

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
        args["aggregations"] = ["min", "max", "mean", "median", "count", "sum"]
        args["key"] = "c"

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [None],
            "b": [2],
            "c": ["A"],
        }
        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)
        # ensure none column is numeric type
        single_row_df = (
            nw.from_native(single_row_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
            )
            .to_native()
        )

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(single_row_df)

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": [None],
            "b": [2],
            "c": ["A"],
            "a_min": [None],
            "a_max": [None],
            "a_mean": [None],
            "a_median": [None],
            "a_count": [0],
            "a_sum": [0],
            "b_min": [2],
            "b_max": [2],
            "b_mean": [2.0],
            "b_median": [2.0],
            "b_count": [1],
            "b_sum": [2],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        # ensure none columns are numeric type
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col(col).cast(nw.Float64)
                for col in ["a", "a_min", "a_max", "a_mean", "a_median", "a_sum"]
            )
            .to_native()
        )

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                *[
                    # counts in polars are set to unsigned int
                    nw.col(col).cast(nw.UInt32)
                    for col in ["a_count", "b_count"]
                ],
            ).to_native()

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
        args["aggregations"] = ["min", "max", "mean", "median", "count", "sum"]
        args["key"] = "c"

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [1, None, 3, None, 8],
            "b": [None, 3, None, 5, 9],
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
            "a_min": [1, None, 1, None, 1],
            "a_max": [8, None, 8, None, 8],
            "a_mean": [4.0, None, 4.0, None, 4.0],
            "a_median": [3.0, None, 3.0, None, 3.0],
            "a_count": [3, 0, 3, 0, 3],
            "a_sum": [12.0, 0.0, 12.0, 0.0, 12.0],
            "b_min": [9.0, 3.0, 9.0, 3.0, 9.0],
            "b_max": [9.0, 5.0, 9.0, 5.0, 9.0],
            "b_mean": [9.0, 4.0, 9.0, 4.0, 9.0],
            "b_median": [9.0, 4.0, 9.0, 4.0, 9.0],
            "b_count": [1, 2, 1, 2, 1],
            "b_sum": [9.0, 8.0, 9.0, 8.0, 9.0],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                [
                    nw.col("a_count").cast(nw.UInt32),
                    nw.col("b_count").cast(nw.UInt32),
                ],
            ).to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
