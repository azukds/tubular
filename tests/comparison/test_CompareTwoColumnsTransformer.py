import copy

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tubular.comparison import ConditionEnum


def create_compare_test_df(library="pandas"):
    """Create a test dataframe for CompareTwoColumnsTransformer tests."""
    df_dict = {
        "a": [1, 2, 3, None, 4],
        "b": [3, 2, 1, 5, None],
    }
    return u.dataframe_init_dispatch(df_dict, library=library)


class TestCompareTwoColumnsTransformerInit:
    """Tests for the initialization of CompareTwoColumnsTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CompareTwoColumnsTransformer"

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

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_raises_error_on_non_numeric_column_type(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method raises TypeError if columns are not numeric."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["condition"] = ConditionEnum.GREATER_THAN.value

        # DataFrame with non-numeric types
        df_dict = {
            "a": ["x", "y", "z"],  # String type
            "b": ["a", "b", "c"],  # String type
        }
        df = u.dataframe_init_dispatch(df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            TypeError,
            match=r"Columns must be of a numeric type, but the following are not: \['a', 'b'\]",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_raises_error_on_mixed_column_types(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method raises TypeError if columns have mixed types."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["condition"] = ConditionEnum.LESS_THAN.value

        # Create a DataFrame with mixed types
        df_dict = {
            "a": [1, 2, 3],  # Int type
            "b": ["a", "b", "c"],  # String type
        }
        df = u.dataframe_init_dispatch(df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        with pytest.raises(
            TypeError,
            match=r"Columns must be of a numeric type, but the following are not: \['b'\]",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("condition", "expected_result"),
        [
            (ConditionEnum.GREATER_THAN, [False, False, True, None, None]),
            (ConditionEnum.LESS_THAN, [True, False, False, None, None]),
            (ConditionEnum.EQUAL_TO, [False, True, False, None, None]),
            (ConditionEnum.NOT_EQUAL_TO, [True, False, True, None, None]),
        ],
    )
    def test_transform_basic_case_outputs(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        lazy,
        condition,
        expected_result,
    ):
        """Test transform method performs comparison correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["condition"] = condition.value

        df = create_compare_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df, lazy))

        # Expected output for basic comparison
        expected_data = {
            "a": [1, 2, 3, None, 4],
            "b": [3, 2, 1, 5, None],
            f"a{condition.value}b": expected_result,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        expected_df = nw.from_native(expected_df)

        expected_df = expected_df.with_columns(
            nw.maybe_convert_dtypes(expected_df[f"a{condition.value}b"]),
        ).to_native()

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("condition", "a_value", "b_value", "expected_result"),
        [
            (ConditionEnum.GREATER_THAN, 1, 3, [False]),
            (ConditionEnum.LESS_THAN, 1, 3, [True]),
            (ConditionEnum.EQUAL_TO, 1, 1, [True]),
            (ConditionEnum.NOT_EQUAL_TO, 1, 1, [False]),
        ],
    )
    def test_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
        lazy,
        condition,
        a_value,
        b_value,
        expected_result,
    ):
        """Test transform method performs comparison correctly on single-row inputs."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["condition"] = condition.value

        # Create a single-row dataframe
        df_dict = {"a": [a_value], "b": [b_value]}
        df = u.dataframe_init_dispatch(df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        # Handle JSON serialization/deserialization
        transformer = u._handle_from_json(transformer, from_json)
        transformed_df = transformer.transform(u._convert_to_lazy(df, lazy))

        # Expected output for single-row comparison
        expected_data = {
            "a": [a_value],
            "b": [b_value],
            f"a{condition.value}b": expected_result,
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        expected_df = nw.from_native(expected_df)

        expected_df = expected_df.with_columns(
            nw.maybe_convert_dtypes(expected_df[f"a{condition.value}b"]),
        ).to_native()

        u.assert_frame_equal_dispatch(
            u._collect_frame(transformed_df, lazy),
            expected_df,
        )
