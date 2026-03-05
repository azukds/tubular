from copy import deepcopy

import narwhals as nw
import numpy as np
import pytest

from tests.test_data import create_df_2
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.mixins import WeightColumnMixin


class TestCreateUnitWeightsColumn:
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_new_column_output(
        self,
        library,
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


class TestCheckWeightsColumn:
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


class TestGetValidWeightsFilterExpr:
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_filter_expression_results(self, library):
        "test that rows with bad weights are filtered as expected by expression"

        obj = WeightColumnMixin()

        df_dict = {
            "weights": [-1.0, np.inf, -np.inf, None, np.nan, 1.0],
            "column": ["a", "b", "c", "d", "e", "f"],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)

        # take the only row with valid weights
        expected_df = df.clone()[[5]]

        filter_expr = obj.get_valid_weights_filter_expr(weights_column="weights")

        df = df.filter(filter_expr)

        assert_frame_equal_dispatch(df.to_native(), expected_df.to_native())

    @pytest.mark.parametrize("verbose", [True, False])
    def test_warning(self, verbose, recwarn):
        "test expected warning is given (depending on verbose)"

        obj = WeightColumnMixin()

        obj.get_valid_weights_filter_expr(weights_column="weights", verbose=verbose)

        if not verbose:
            assert len(recwarn) == 0

        else:
            assert len(recwarn) == 1
            assert (
                str(recwarn[0].message)
                == "Weights must be strictly positive, non-null, and finite - rows failing this will be filtered out."
            )
