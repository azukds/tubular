import pytest
import copy
from tests import utils as u
from beartype.roar import BeartypeCallHintParamViolation
import narwhals as nw

from tests.base_tests import ColumnStrListInitTests


def create_when_then_test_df(library="pandas"):
    """Create a test dataframe for WhenThenOtherwiseTransformer tests."""
    df_dict = {
        "a": [10, 20, 30],
        "b": [40, 50, 60],
        "condition_col": [True, False, True],
        "update_col": [100, 200, 300],
    }
    return u.dataframe_init_dispatch(df_dict, library=library)



class TestWhenThenOtherwiseTransformerInit(ColumnStrListInitTests):
    """Tests for init method in WhenThenOtherwiseTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "WhenThenOtherwiseTransformer"

    @pytest.mark.parametrize(
        "columns, when_column, then_column",
        [
            ([], "condition_col", "update_col"), 
            (["a"], "condition_col", "update_col"),
            (None, "condition_col", "update_col"),
            ("a", "condition_col", "update_col"),
            (123, "condition_col", "update_col"),
            (["a", "b"], "condition_col", "update_col"),
            (["a", "b", "c"], "condition_col", "update_col"),
        ],
    )
    def test_errors_if_invalid_columns(
        self,
        columns,
        when_column,
        then_column,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns
        args["when_column"] = when_column
        args["then_column"] = then_column
        if columns in ([], None, "a", 123):
            with pytest.raises(BeartypeCallHintParamViolation):
                uninitialized_transformers[self.transformer_name](**args)
        else:
            transformer = uninitialized_transformers[self.transformer_name](**args)
            assert transformer is not None

    @pytest.mark.parametrize(
        "when_column, then_column",
        [
            (None, "update_col"), 
            ("condition_col", None),
            (123, "update_col"), 
            ("condition_col", 456), 
        ],
    )
    def test_errors_if_invalid_when_then_columns(
        self,
        when_column,
        then_column,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["when_column"] = when_column
        args["then_column"] = then_column
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)



class TestWhenThenOtherwiseTransformerTransform():
    """Tests for transform method in WhenThenOtherwiseTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "WhenThenOtherwiseTransformer"

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
        """Test transform method performs conditional updates correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["when_column"] = "condition_col"
        args["then_column"] = "update_col"
        
        df = create_when_then_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df, lazy))

        # Expected output for basic conditional update
        expected_data = {
            "a": [100, 20, 300],
            "b": [100, 50, 300],
            "condition_col": [True, False, True],
            "update_col": [100, 200, 300],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "a_values, b_values, condition_values, update_values, expected_a, expected_b",
        [
            ([10], [40], [True], [100], [100], [100]),
            ([10], [40], [False], [100], [10], [40]),
            ([None], [40], [True], [100], [100], [100]),
            ([10], [None], [False], [100], [10], [None]),
            ([None], [None], [True], [100], [100], [100]),
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
        condition_values,
        update_values,
        expected_a,
        expected_b,
        lazy,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["when_column"] = "condition_col"
        args["then_column"] = "update_col"
        
        single_row_df_dict = {
            "a": a_values,
            "b": b_values,
            "condition_col": condition_values,
            "update_col": update_values,
        }
        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)
        single_row_df = (
            nw.from_native(single_row_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("condition_col").cast(nw.Boolean),
                nw.col("update_col").cast(nw.Float64),
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
            "a": expected_a,
            "b": expected_b,
            "condition_col": condition_values,
            "update_col": update_values,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("condition_col").cast(nw.Boolean),
                nw.col("update_col").cast(nw.Float64),
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
        args["when_column"] = "condition_col"
        args["then_column"] = "update_col"
        
        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [10, None, 30],
            "b": [40, 50, None],
            "condition_col": [True, False, None],
            "update_col": [100, 200, 300],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        df_with_nulls = (
            nw.from_native(df_with_nulls)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("condition_col").cast(nw.Boolean),
                nw.col("update_col").cast(nw.Float64),
            )
            .to_native()
        )

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df_with_nulls, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df_with_nulls, lazy))

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [100, None, 30],
            "b": [100, 50, None],
            "condition_col": [True, False, None],
            "update_col": [100, 200, 300],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        expected_df = (
            nw.from_native(expected_df)
            .with_columns(
                nw.col("a").cast(nw.Float64),
                nw.col("b").cast(nw.Float64),
                nw.col("condition_col").cast(nw.Boolean),
                nw.col("update_col").cast(nw.Float64),
            )
            .to_native()
        )

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )