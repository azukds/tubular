import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
import tests.utils as u
from tests.base_tests import DropOriginalInitMixinTests, DropOriginalTransformMixinTests
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerFitTests,
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import OneDKmeansTransformer


class TestInit(
    BaseNumericTransformerInitTests,
    DropOriginalInitMixinTests,
):
    """Tests for OneDKmeansTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    def test_clone(self):
        pass

    def test_acolumns_type_error_if_not_str_or_len1_list(self):
        """Test that an exception is raised if kmeans_kwargs is not a dict."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneDKmeansTransformer(columns=["b", "c"], new_column_name="a")


class TestFit(BaseNumericTransformerFitTests):
    """Tests for OneDKmeansTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    # This is dealt by @deartype
    def test_unexpected_kwarg_error(self):
        pass

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator"),
        [
            (d.create_df_9),  # int with None
            (d.create_bool_and_float_df),  # float with np.nan
            (d.create_df_with_none_and_nan_cols),  # all np.nan
        ],
    )
    def test_x_nans_value_type_error(
        self,
        library,
        df_generator,
    ):
        """Test that an exception is raised if X contains Nan or None."""
        with pytest.raises(
            ValueError,
            match=r"OneDKmeansTransformer: X should not contain missing values.",
        ):
            df = df_generator(library=library)
            OneDKmeansTransformer(
                new_column_name="b",
                columns="a",
            ).fit(X=df)


# Create test data
def create_numeric_df_1(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5],
        "b": [43, 77, 61, 29, 84, 29, 24, 40, 84, 96],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_1(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5],
        "b": [43, 77, 61, 29, 84, 29, 24, 40, 84, 96],
        "new": [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_1_drop(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5],
        "new": [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_numeric_df_2(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5, 10, 12, 4, 16, 17],
        "b": [43, -77, -61, 29, 84, 29, -24, 40, 84, -96, 10, -4, 15, -12, 15],
        "c": [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "c",
            "b",
            "a",
            "c",
            "a",
            "c",
            "a",
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_2(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5, 10, 12, 4, 16, 17],
        "b": [43, -77, -61, 29, 84, 29, -24, 40, 84, -96, 10, -4, 15, -12, 15],
        "c": [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "c",
            "b",
            "a",
            "c",
            "a",
            "c",
            "a",
        ],
        "new": [3, 0, 0, 3, 4, 3, 1, 3, 4, 0, 2, 1, 2, 1, 2],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_2_drop(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5, 10, 12, 4, 16, 17],
        "c": [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "c",
            "b",
            "a",
            "c",
            "a",
            "c",
            "a",
        ],
        "new": [3, 0, 0, 3, 4, 3, 1, 3, 4, 0, 2, 1, 2, 1, 2],
    }

    return u.dataframe_init_dispatch(df_dict, library)


class TestTransform(
    BaseNumericTransformerTransformTests,
    DropOriginalTransformMixinTests,
):
    """Tests for OneDKmeansTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_1(library="pandas"),
                expected_numeric_df_1(library="pandas"),
            ),
            (
                create_numeric_df_1(library="polars"),
                expected_numeric_df_1(library="polars"),
            ),
        ],
    )
    def test_expected_output_without_drop(self, df, expected):
        """Test that the output is expected from transform, when there are no negative numbers and dont drop original"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=2,
            new_column_name="new",
            drop_original=False,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_1(library="pandas"),
                expected_numeric_df_1_drop(library="pandas"),
            ),
            (
                create_numeric_df_1(library="polars"),
                expected_numeric_df_1_drop(library="polars"),
            ),
        ],
    )
    def test_expected_output_with_drop(self, df, expected):
        """Test that the output is expected from transform, when there are no negative numbers and drop original"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=2,
            new_column_name="new",
            drop_original=True,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_2(library="pandas"),
                expected_numeric_df_2(library="pandas"),
            ),
            (
                create_numeric_df_2(library="polars"),
                expected_numeric_df_2(library="polars"),
            ),
        ],
    )
    def test_expected_output_without_drop_negatives(self, df, expected):
        """Test that the output is expected from transform, when there are negative numbers and dont drop original"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=5,
            new_column_name="new",
            drop_original=False,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_2(library="pandas"),
                expected_numeric_df_2_drop(library="pandas"),
            ),
            (
                create_numeric_df_2(library="polars"),
                expected_numeric_df_2_drop(library="polars"),
            ),
        ],
    )
    def test_expected_output_with_drop_negatives(self, df, expected):
        """Test that the output is expected from transform, when there are negative numbers and drop original"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=5,
            new_column_name="new",
            drop_original=True,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)
