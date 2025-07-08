import narwhals as nw
import numpy as np
import pytest

from tests.test_data import create_df_2
from tubular.mixins import WeightColumnMixin


class TestCheckWeightsColumn:
    @pytest.mark.parametrize(
        "library",
        [
            "pandas",
            "polars",
        ],
    )
    @pytest.mark.parametrize(
        "bad_weight_value, expected_message",
        [
            (None, "weight column must be non-null"),
            (np.inf, "weight column must not contain infinite values."),
            (-np.inf, "weight column must be positive"),
            (-1, "weight column must be positive"),
        ],
    )
    def test_bad_values_in_weights_error(
        self,
        bad_weight_value,
        expected_message,
        library,
    ):
        """Test that an exception is raised if there are negative/nan/inf values in sample_weight."""

        df = create_df_2(library=library)

        obj = WeightColumnMixin()

        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)

        weight_column = "weight_column"

        df = df.with_columns(
            nw.new_series(
                weight_column,
                [*[bad_weight_value], *np.arange(2, len(df) + 1)],
                backend=native_backend,
            ),
        )

        df = nw.to_native(df)

        with pytest.raises(ValueError, match=expected_message):
            obj.check_weights_column(df, weight_column)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_weight_col_non_numeric(
        self,
        library,
    ):
        """Test an error is raised if weight is not numeric."""

        obj = WeightColumnMixin()

        df = create_df_2(library=library)
        df = nw.from_native(df)

        weight_column = "weight_column"
        error = r"weight column must be numeric."
        df = df.with_columns(nw.lit("a").alias(weight_column))
        df = nw.to_native(df)

        with pytest.raises(
            ValueError,
            match=error,
        ):
            # using check_weights_column method to test correct error is raised for transformers that use weights

            obj.check_weights_column(df, weight_column)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_weight_not_in_X_error(
        self,
        library,
    ):
        """Test an error is raised if weight is not in X"""

        obj = WeightColumnMixin()

        df = create_df_2(library=library)

        weight_column = "weight_column"
        error = rf"weight col \({weight_column}\) is not present in columns of data"

        with pytest.raises(
            ValueError,
            match=error,
        ):
            # using check_weights_column method to test correct error is raised for transformers that use weights

            obj.check_weights_column(df, weight_column)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_zero_total_weight_error(
        self,
        library,
    ):
        """Test that an exception is raised if the total sample weights are 0."""

        obj = WeightColumnMixin()

        weight_column = "weight_column"

        df = create_df_2(library=library)

        df = nw.from_native(df)
        df = df.with_columns(nw.lit(0).alias(weight_column))
        df = nw.to_native(df)

        with pytest.raises(
            ValueError,
            match="total sample weights are not greater than 0",
        ):
            obj.check_weights_column(df, weight_column)
