import re

import pytest

from tests.test_data import (
    create_bool_and_float_df,
    create_df_2,
    create_df_with_none_and_nan_cols,
    create_is_between_dates_df_1,
)
from tubular.mixins import CheckNumericMixin


class TestCheckNumericMixin:
    "tests for CheckNumericMixin class"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator", "bad_cols"),
        [
            (create_df_2, ["b", "c"]),  # str
            (create_is_between_dates_df_1, ["a"]),  # datetime
            (create_bool_and_float_df, ["b"]),  # bool
            (create_df_with_none_and_nan_cols, ["b"]),  # None
        ],
    )
    def test_check_numeric_columns_errors(self, library, df_generator, bad_cols):
        "test check_numeric_columns method raises appropriate error"

        df = df_generator(library=library)

        obj = CheckNumericMixin()

        # this object is generally wrapped in a transformer with a .columns attr, set this here
        obj.columns = bad_cols

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"CheckNumericMixin: The following columns are not numeric in X; {bad_cols}",
            ),
        ):
            obj.check_numeric_columns(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator", "cols"),
        [
            (create_df_2, ["a"]),  # int
            (create_bool_and_float_df, ["a"]),  # float
            (create_df_with_none_and_nan_cols, ["a"]),  # nan
        ],
    )
    def test_check_numeric_columns_passes(self, library, df_generator, cols):
        "test check_numeric_columns method passes for numeric columns"

        df = df_generator(library=library)

        obj = CheckNumericMixin()

        # this object is generally wrapped in a transformer with a .columns attr, set this here
        obj.columns = cols

        obj.check_numeric_columns(df)
