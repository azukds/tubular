import pytest

from tests.test_data import create_df_1
from tests.utils import assert_frame_equal_dispatch
from tubular.mixins import DropOriginalMixin


class TestDropOriginalColumn:
    "tests for DropOriginalMixin.drop_original_column"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("drop_original", [True, False])
    def test_drop_original_arg_handling(
        self,
        library,
        drop_original,
    ):
        """Test transformer drops/keeps original columns when specified/not specified."""

        df = create_df_1(library=library)

        obj = DropOriginalMixin()

        columns = list(df.columns)

        df_transformed = obj.drop_original_column(
            df,
            drop_original=drop_original,
            columns=columns,
        )

        remaining_cols = df_transformed.columns

        if drop_original:
            for col in columns:
                assert col not in remaining_cols, "original columns not dropped"

        else:
            for col in columns:
                assert col in remaining_cols, "original columns not kept"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_other_columns_not_modified(
        self,
        library,
    ):
        """Test transformer does not modify unspecified columns."""

        df = create_df_1(library=library)

        obj = DropOriginalMixin()

        drop_original = False

        columns = ["a"]

        df_transformed = obj.drop_original_column(
            df,
            drop_original=drop_original,
            columns=columns,
        )

        other_columns = list(set(df.columns) - set(columns))

        assert_frame_equal_dispatch(df[other_columns], df_transformed[other_columns])
