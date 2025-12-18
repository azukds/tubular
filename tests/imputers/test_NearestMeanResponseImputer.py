import narwhals as nw
import numpy as np
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    ReturnNativeTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.imputers import NearestMeanResponseImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_null_values_in_response_error(self, library):
        """Test an error is raised if the response column contains null entries."""
        df = d.create_df_3(library=library)

        transformer = NearestMeanResponseImputer(columns=["b"])

        with pytest.raises(
            ValueError,
            match="NearestMeanResponseImputer: y has 1 null values",
        ):
            transformer.fit(df, df["a"])

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_columns_with_no_nulls_warning(self, library):
        """Test a warning is raised if a non-response column contains no nulls."""
        df = d.create_numeric_df_1(library=library)

        transformer = NearestMeanResponseImputer(columns=["c"])

        with pytest.warns(
            UserWarning,
            match="NearestMeanResponseImputer: Column c has no missing values, this transformer will have no effect for this column.",
        ):
            transformer.fit(df, df["c"])

        expected_impute_values = {"c": None}
        assert transformer.impute_values_ == expected_impute_values, (
            f"impute_values_ attr not as expected, expected {expected_impute_values} but got {transformer.impute_values_}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values(self, library):
        """Test that the nearest response values learnt during fit are expected."""
        df = d.create_numeric_df_2(library=library)

        transformer = NearestMeanResponseImputer(columns=["b", "c"])

        transformer.fit(df, df["a"])

        assert transformer.impute_values_ == {
            "b": np.float64(3),
            "c": np.float64(2),
        }, "impute_values_ attribute"


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("fit_col", "transform_col"),
        [
            # try a few types, with and without nulls in transform col
            ([1, 2, 3], [1.0, np.nan, np.nan]),
            ([4, 5, 6], [7, 8, 9]),
            (["a", "b", "c"], ["a", None, "d"]),
            (["c", "d", "e"], ["f", "g", "h"]),
            ([4.0, 5.0, 6.0], [8.0, np.nan, 6.0]),
            ([1.0, 2.0, 3.0], [4.0, 3.0, 2.0]),
            ([True, False, False], [True, True, None]),
            ([True, False, True], [True, False, True]),
        ],
    )
    def test_no_effect_when_fit_on_null_free_col(self, fit_col, transform_col, library):
        "test that when transformer fits on a col with no nulls, transform has no effect"

        df_fit_dict = {
            "a": fit_col,
            "b": [1] * len(fit_col),
        }

        df_fit = dataframe_init_dispatch(df_fit_dict, library=library)

        df_transform_dict = {
            "a": transform_col,
        }

        df_transform = dataframe_init_dispatch(df_transform_dict, library=library)

        transformer = NearestMeanResponseImputer(columns=["a"])

        transformer.fit(df_fit, df_fit["b"])

        df_transform = nw.from_native(df_transform)

        expected_output = df_transform.clone().to_native()

        df_transform = nw.to_native(df_transform)

        output = transformer.transform(df_transform)

        assert_frame_equal_dispatch(output, expected_output)
