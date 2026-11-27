import copy
import re

import narwhals as nw
import pytest

from tests.aggregations.test_BaseAggregationTransformer import (
    TestBaseAggregationTransformerInit,
    TestBaseAggregationTransformerTransform,
)
from tests.base_tests import (
    EmptyColumnsFitTransformPassTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
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

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("aggregations", "expected_data"),
        [
            (
                ["min", "max", "mean", "sum"],
                {
                    "a": [1, 2, 3, 4, 0],
                    "b": [2, 3, 4, 5, 0],
                    "c": [True, False, False, True, True],
                    "a_b_c_min": [1, 0, 0, 1, 0],
                    "a_b_c_max": [2, 3, 4, 5, 1],
                    "a_b_c_mean": [4 / 3, 5 / 3, 7 / 3, 10 / 3, 1 / 3],
                    "a_b_c_sum": [4, 5, 7, 10, 1],
                },
            ),
        ],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    def test_transform_basic_case_outputs(
        self,
        library,
        aggregations,
        expected_data,
        minimal_attribute_dict,
        uninitialized_transformers,
        lazy,
        from_json,
    ):
        """Test transform method aggregates rows correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b", "c"]
        args["aggregations"] = aggregations
        args["bool_columns"] = ["c"]

        df_dict = {
            "a": [1, 2, 3, 4, 0],
            "b": [2, 3, 4, 5, 0],
            "c": [True, False, False, True, True],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df, lazy):
            return
        transformer = _handle_from_json(transformer, from_json=from_json)
        transformed_df = transformer.transform(_convert_to_lazy(df, lazy))

        # Create expected DataFrame using the library parameter
        expected_df = dataframe_init_dispatch(expected_data, library)

        # Compare the transformed DataFrame with the expected DataFrame using the dispatch function
        assert_frame_equal_dispatch(
            _collect_frame(transformed_df, lazy),
            expected_df,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        lazy,
        from_json,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b", "c"]
        args["aggregations"] = ["min", "max", "mean", "sum"]
        args["bool_columns"] = ["c"]

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [None],
            "b": [2],
            "c": [True],
            "d": ["A"],
        }
        single_row_df = dataframe_init_dispatch(single_row_df_dict, library)

        # cast none row to numeric type
        single_row_df = nw.from_native(single_row_df)
        single_row_df = single_row_df.with_columns(
            nw.col("a").cast(nw.Float64),
        ).to_native()

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, single_row_df, lazy):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)
        transformed_df = transformer.transform(_convert_to_lazy(single_row_df, lazy))

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": [None],
            "b": [2],
            "c": [True],
            "d": ["A"],
            "a_b_c_min": [1.0],
            "a_b_c_max": [2.0],
            "a_b_c_mean": [1.5],
            "a_b_c_sum": [3.0],
        }
        expected_df = dataframe_init_dispatch(expected_data, library)
        # ensure none columns are numeric type
        expected_df = (
            nw.from_native(expected_df)
            .with_columns(nw.col(col).cast(nw.Float64) for col in ["a"])
            .to_native()
        )

        assert_frame_equal_dispatch(
            _collect_frame(transformed_df, lazy),
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
        lazy,
        from_json,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b", "c"]
        args["aggregations"] = ["min", "max", "mean", "sum"]
        args["bool_columns"] = ["c"]

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [1.0, None, 3.0, None, 8.0],
            "b": [None, 3.0, None, 5.0, 9.0],
            "c": [None, True, None, False, False],
            "d": ["A", "B", "A", "B", "A"],
        }
        df_with_nulls = dataframe_init_dispatch(df_with_nulls_dict, library)
        df_with_nulls = nw.to_native(
            nw.maybe_convert_dtypes(nw.from_native(df_with_nulls))
        )

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df_with_nulls, lazy):
            return
        transformer = _handle_from_json(transformer, from_json=from_json)
        transformed_df = transformer.transform(_convert_to_lazy(df_with_nulls, lazy))

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [1, None, 3, None, 8],
            "b": [None, 3, None, 5, 9],
            "c": [None, True, None, False, False],
            "d": ["A", "B", "A", "B", "A"],
            "a_b_c_min": [1.0, 1.0, 3.0, 0.0, 0.0],
            "a_b_c_max": [1.0, 3.0, 3.0, 5.0, 9.0],
            "a_b_c_mean": [1.0, 2.0, 3.0, 2.5, 17 / 3],
            "a_b_c_sum": [1.0, 4.0, 3.0, 5.0, 17.0],
        }
        expected_df = dataframe_init_dispatch(expected_data, library)

        assert_frame_equal_dispatch(
            _collect_frame(transformed_df, lazy),
            nw.to_native(nw.maybe_convert_dtypes(nw.from_native(expected_df))),
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("bool_columns", [["a"], ["bla"]])
    def test_bool_column_not_exist_error(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        lazy,
        from_json,
        bool_columns: list[str],
    ):
        """Test transform method raises error when non-bool column in bool_columns."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a"]
        args["aggregations"] = ["min", "max", "mean", "sum"]
        args["bool_columns"] = bool_columns

        df_dict = {"a": [1, 2, 3, 4, 0]}
        df = dataframe_init_dispatch(df_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df, lazy):
            return
        transformer = _handle_from_json(transformer, from_json=from_json)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "AggregateColumnsOverRowTransformer: invalid bool_columns; remove "
                f"non-Boolean or unselected columns {bool_columns}"
            )
        ):
            transformer.transform(_convert_to_lazy(df, lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_bool_column_not_provided_error(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        lazy,
        from_json,
    ):
        """Test transform method raises error when bool column not in bool_columns."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a"]
        args["aggregations"] = ["min", "max", "mean", "sum"]
        args["bool_columns"] = None

        df_dict = {"a": [True, False, False, True, True]}
        df = dataframe_init_dispatch(df_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df, lazy):
            return
        transformer = _handle_from_json(transformer, from_json=from_json)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "AggregateColumnsOverRowTransformer: invalid bool_columns; add "
                "Boolean columns ['a']"
            )
        ):
            transformer.transform(_convert_to_lazy(df, lazy))


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyColumnsFitTransformPassTests,
):
    """
    Class to run tests for AggregateColumnsOverRowTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateColumnsOverRowTransformer"
