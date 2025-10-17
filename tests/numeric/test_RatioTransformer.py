import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)


def create_ratio_test_df(library="pandas"):
    """Create a test dataframe for RatioTransformer tests."""
    df_dict = {
        "a": [100, 200, 300, 400],
        "b": [80, 150, 200, 0],
    }
    return u.dataframe_init_dispatch(df_dict, library)


class TestRatioTransformerInit(BaseNumericTransformerInitTests):
    """Tests for init method in RatioTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RatioTransformer"

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

    @pytest.mark.parametrize("return_dtype", ["InvalidType", None])
    def test_return_dtype_arg_errors(
        self,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["return_dtype"] = return_dtype
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestRatioTransformerTransform(BaseNumericTransformerTransformTests):
    """Tests for transform method in RatioTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RatioTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_transform_basic_case_outputs(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test transform method performs division correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

        df = create_ratio_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(df)

        # Expected output for basic division
        expected_data = {
            "a": [100, 200, 300, 400],
            "b": [80, 150, 200, 0],
            "a_divided_by_b": [1.25, 1.333333, 1.5, None],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "a_values, b_values, expected_division",
        [
            ([100], [80], [1.25]),
            ([0], [80], [0]),
            ([100], [0], [None]),
            ([None], [10], [None]),
            ([10], [None], [None]),
        ],
    )
    def test_single_row(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        a_values,
        b_values,
        expected_division,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

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
            "a_divided_by_b": expected_division,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
            )
            .to_native()
        )

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_with_nulls(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

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
            "a_divided_by_b": [1.25, None, None],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
