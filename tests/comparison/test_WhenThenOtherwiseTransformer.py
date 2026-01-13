import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.base_tests import ColumnStrListInitTests


def create_when_then_test_df(library="pandas"):
    """Create a test dataframe for WhenThenOtherwiseTransformer tests."""
    df_dict = {
        "a": [10, 20, 30, None, 60, 70],
        "b": [40, 50, 60, 50, 70, 80],
        "condition_col": [True, False, True, True, None, True],
        "update_col": [100, 200, 300, 200, 300, None],
    }
    return (
        nw.from_native(u.dataframe_init_dispatch(df_dict, library=library))
        .with_columns(
            nw.col("a").cast(nw.Float64),
            nw.col("b").cast(nw.Float64),
            nw.col("condition_col").cast(nw.Boolean),
            nw.col("update_col").cast(nw.Float64),
        )
        .to_native()
    )


def create_invalid_type_df(library="pandas"):
    """Create a test dataframe with invalid types for testing."""
    df_dict = {
        "a": [1, 2, 3],  # Int type
        "b": [4.0, 5.0, 6.0],  # Float type
        "condition_col": [1, 0, 1],  # Int type, should be Boolean
        "update_col": [10, 20, 30],  # Int type
    }
    return (
        nw.from_native(u.dataframe_init_dispatch(df_dict, library=library))
        .with_columns(
            nw.col("a").cast(nw.Int64),
            nw.col("b").cast(nw.Float64),
            nw.col("condition_col").cast(nw.Int64),  # Incorrect type
            nw.col("update_col").cast(nw.Int64),
        )
        .to_native()
    )


def create_mismatched_type_df(library="pandas"):
    """Create a test dataframe with mismatched types for testing."""
    df_dict = {
        "a": [1, 2, 3],  # Int type
        "b": [4, 5, 6],  # Int type
        "condition_col": [True, False, True],  # Correct Boolean type
        "update_col": [10.0, 20.0, 30.0],  # Float type
    }
    return (
        nw.from_native(u.dataframe_init_dispatch(df_dict, library=library))
        .with_columns(
            nw.col("a").cast(nw.Int64),
            nw.col("b").cast(nw.Int64),
            nw.col("condition_col").cast(nw.Boolean),
            nw.col("update_col").cast(nw.Float64),  # Mismatched type
        )
        .to_native()
    )


class TestWhenThenOtherwiseTransformerInit(ColumnStrListInitTests):
    """Tests for init method in WhenThenOtherwiseTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "WhenThenOtherwiseTransformer"

    @pytest.mark.parametrize(
        ("when_column", "then_column"),
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


class TestWhenThenOtherwiseTransformerTransform:
    """Tests for transform method in WhenThenOtherwiseTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "WhenThenOtherwiseTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_raises_error_on_invalid_condition_column_type(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method raises TypeError if condition column is not Boolean."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["when_column"] = "condition_col"
        args["then_column"] = "update_col"

        df = create_invalid_type_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            TypeError, match=r"The column 'condition_col' must be of type Boolean."
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_raises_error_on_mismatched_column_types(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method raises TypeError if columns have mismatched types."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["when_column"] = "condition_col"
        args["then_column"] = "update_col"

        df = create_mismatched_type_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            TypeError,
            match=r"All columns in .* must be of the same type as 'update_col'.",
        ):
            transformer.transform(df)

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
            "a": [100, 20, 300, 200, 60, None],
            "b": [100, 50, 300, 200, 70, None],
            "condition_col": [True, False, True, True, None, True],
            "update_col": [100, 200, 300, 200, 300, None],
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
    @pytest.mark.parametrize(
        (
            "a_values",
            "b_values",
            "condition_values",
            "update_values",
            "expected_a",
            "expected_b",
        ),
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
