import narwhals as nw
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.utils import (
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.imputers import NullIndicator


class TestInit(ColumnStrListInitTests):
    """Tests for NullIndicator.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"


class TestTransform(GenericTransformTests, ReturnNativeTests):
    """Tests for NullIndicator.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"

    @pytest.fixture()
    def expected_df_1(self, request):
        """Expected output for test_null_indicator_columns_correct."""
        library = request.param

        df_dict1 = {
            "a": [1, 2, None, 4, None, 6],
            "b": [None, 5, 4, 3, 2, 1],
            "c": [3, 2, 1, 4, 5, 6],
            "b_nulls": [1, 0, 0, 0, 0, 0],
            "c_nulls": [0, 0, 0, 0, 0, 0],
        }

        df1 = dataframe_init_dispatch(dataframe_dict=df_dict1, library=library)

        narwhals_df = nw.from_native(df1)

        # Convert adjusted expected columns to Boolean
        for col in ["b_nulls", "c_nulls"]:
            narwhals_df = narwhals_df.with_columns(
                narwhals_df[col].cast(nw.Boolean),
            )

        return narwhals_df.to_native()

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_1"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_1"],
    )
    def test_null_indicator_columns_correct(self, expected_df_1, library, from_json):
        """Test that the created indicator column is correct - and unrelated columns are unchanged."""
        df = d.create_df_9(library=library)

        columns = ["b", "c"]
        transformer = NullIndicator(columns=columns)

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(df)

        # Convert both DataFrames to a common format using Narwhals
        df_transformed_common = nw.from_native(df_transformed)
        expected_df_1_common = nw.from_native(expected_df_1)

        # Check outcomes for single rows
        for i in range(len(df_transformed_common)):
            df_transformed_row = df_transformed_common[[i]].to_native()
            df_expected_row = expected_df_1_common[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

        # Check whole dataframes
        assert_frame_equal_dispatch(
            df_transformed_common.to_native(),
            expected_df_1_common.to_native(),
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"
