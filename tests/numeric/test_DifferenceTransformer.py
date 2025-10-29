import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)


def create_difference_test_df(library="pandas"):
    """Create a test dataframe for DifferenceTransformer tests."""
    df_dict = {
        "a": [100, 200, 300],
        "b": [80, 150, 200],
    }
    return u.dataframe_init_dispatch(df_dict, library)


class TestDifferenceTransformerInit(BaseNumericTransformerInitTests):
    """Tests for init method in DifferenceTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DifferenceTransformer"

    @pytest.mark.parametrize("columns", (["a"], ["a", "b", "c"], None, "a"))
    def test_errors_if_not_two_columns(
        self,
        columns,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestDifferenceTransformerTransform(BaseNumericTransformerTransformTests):
    """Tests for transform method in DifferenceTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DifferenceTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_transform_basic_case_outputs(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test transform method performs subtraction correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]

        df = create_difference_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(df)

        # Expected output for basic subtraction
        expected_data = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
            "a_minus_b": [20, 50, 100],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "a_values, b_values, expected_value",
        [
            ([100], [80], 20),
            ([0], [80], [-80]),
            ([100], [0], [100]),
            ([None], [10], [None]),
            ([10], [None], [None]),
        ],
    )
    def test_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        a_values,
        b_values,
        expected_value,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]

        single_row_df_dict = {
            "a": a_values,
            "b": b_values,
        }

        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)

        single_row_df = (
            nw.from_native(single_row_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
            )
            .to_native()
        )

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(single_row_df)

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": a_values,
            "b": b_values,
            "a_minus_b": expected_value,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a_minus_b").cast(nw.Float64),
            )
            .to_native()
        )

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_with_nulls(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [100, None, 300],
            "b": [80, 150, None],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(df_with_nulls)

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [100, None, 300],
            "b": [80, 150, None],
            "a_minus_b": [20, None, None],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
