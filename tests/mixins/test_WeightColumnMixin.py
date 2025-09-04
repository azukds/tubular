from copy import deepcopy

import narwhals as nw
import numpy as np
import pytest

from tests.test_data import create_df_2
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.mixins import WeightColumnMixin


class TestCreateUnitWeightsColumn:
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("i", [-1, 0, 1, 2, 3, 4])
    def test_new_column_output(
        self,
        library,
        i,
    ):
        """Test unit weights column created as expected"""

        obj = WeightColumnMixin()

        df_dict = {
            "a": [1, 2, 3, 4],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_dict = deepcopy(df_dict)

        expected_new_col = "unit_weights_column"

        expected_dict[expected_new_col] = [1, 1, 1, 1]

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        output, unit_weights_column = obj._create_unit_weights_column(
            df,
            backend=library,
        )

        assert_frame_equal_dispatch(expected, output)

        assert unit_weights_column == "unit_weights_column"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_existing_column_used_if_possible(
        self,
        library,
    ):
        """Test existing unit weights column used if possible"""

        obj = WeightColumnMixin()

        df_dict = {
            "a": [1, 2, 3, 4],
        }

        good_weight_vals = [1, 1, 1, 1]

        df_dict["unit_weights_column"] = good_weight_vals

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_dict = deepcopy(df_dict)

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        output, unit_weights_column = obj._create_unit_weights_column(
            df,
            backend=library,
        )

        assert_frame_equal_dispatch(expected, output)

        assert unit_weights_column == "unit_weights_column"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_errors_if_bad_column_exists(
        self,
        library,
    ):
        """Test that error is raised if unit_weights_column exists but is not all 1"""

        obj = WeightColumnMixin()

        df_dict = {
            "a": [1, 2, 3, 4],
        }

        df_dict["unit_weights_column"] = [1, 2, 1, 1]

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        msg = "Attempting to insert column of unit weights named 'unit_weights_column', but an existing column shares this name and is not all 1, please rename existing column"
        with pytest.raises(
            RuntimeError,
            match=msg,
        ):
            obj._create_unit_weights_column(df, backend=library)


class TestCheckAndSetWeight:
    @pytest.mark.parametrize("weights_column", (0, ["a"], {"a": 10}))
    def test_weight_arg_errors(
        self,
        weights_column,
    ):
        """Test that appropriate errors are throw for bad weight arg."""

        obj = WeightColumnMixin()

        with pytest.raises(
            TypeError,
            match="weights_column should be str or None",
        ):
            obj.check_and_set_weight(weights_column)


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
