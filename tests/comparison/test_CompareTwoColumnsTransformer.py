import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tubular.comparison import ConditionEnum


def create_compare_test_df(library="pandas"):
    """Create a test dataframe for CompareTwoColumnsTransformer tests."""
    df_dict = {
        "a": [1, 2, 3],
        "b": [3, 2, 1],
    }
    return u.dataframe_init_dispatch(df_dict, library=library)


class TestCompareTwoColumnsTransformerInit:
    """Tests for the initialization of CompareTwoColumnsTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CompareTwoColumnsTransformer"

    @pytest.mark.parametrize(
        "columns, condition",
        [
            ([], ConditionEnum.GREATER_THAN),
            (["col1"], ConditionEnum.GREATER_THAN),
            (None, ConditionEnum.GREATER_THAN),
            ("col1", ConditionEnum.GREATER_THAN),
            (123, ConditionEnum.GREATER_THAN),
            (["col1", "col2"], ConditionEnum.GREATER_THAN),
            (["col1", "col2", "col3"], ConditionEnum.GREATER_THAN),
        ],
    )
    def test_errors_if_invalid_columns(
        self,
        columns,
        condition,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns
        #args["condition"] = condition
        if columns in ([], ["col1"], None, "col1", 123, ["col1", "col2", "col3"]):
            with pytest.raises(BeartypeCallHintParamViolation):
                uninitialized_transformers[self.transformer_name](**args)
        else:
            transformer = uninitialized_transformers[self.transformer_name](**args)
            assert transformer is not None

    @pytest.mark.parametrize(
        "condition",
        [
            None,
            123,
            "invalid_condition",
        ],
    )
    def test_errors_if_invalid_condition(
        self,
        condition,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["condition"] = condition
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestCompareTwoColumnsTransformerTransform:
    """Tests for the transform method of CompareTwoColumnsTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CompareTwoColumnsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_transform_basic_case_outputs(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        lazy,
    ):
        """Test transform method performs comparison correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        # args["kwargs"] = {}

        print("Debug: Transformer args:", args)

        df = create_compare_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df, lazy))

        # Expected output for basic comparison
        expected_data = {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
            "a>b": [0, 0, 1],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        transformed_df = (
            nw.from_native(transformed_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "a_values, b_values, expected_result",
        [
            ([1], [3], [0]),
            ([3], [3], [0]),
            ([3], [1], [1]),
            ([None], [1], [0]),
            ([1], [None], [0]),
            ([None], [None], [0]),
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
        expected_result,
        lazy,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        #args["condition"] = ConditionEnum.GREATER_THAN

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

        if u._check_if_skip_test(transformer, single_row_df, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(single_row_df, lazy))

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": a_values,
            "b": b_values,
            "a>b": expected_result,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        # Ensure transformed_df has matching dtypes
        transformed_df = (
            nw.from_native(transformed_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_with_nulls(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        lazy,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        #args["condition"] = ConditionEnum.GREATER_THAN

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [1, None, 3],
            "b": [3, 2, None],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        df_with_nulls = (
            nw.from_native(df_with_nulls)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
            )
            .to_native()
        )

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df_with_nulls, lazy, from_json):
            return

        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df_with_nulls, lazy))

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [1, None, 3],
            "b": [3, 2, None],
            "a>b": [0, 0, 0],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        # Ensure transformed_df has matching dtypes
        transformed_df = (
            nw.from_native(transformed_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("a>b").cast(nw.Int64),
            )
            .to_native()
        )

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )
