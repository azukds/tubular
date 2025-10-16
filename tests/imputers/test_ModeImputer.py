import narwhals as nw
import numpy as np
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
)
from tests.utils import dataframe_init_dispatch
from tubular.imputers import ModeImputer


def input_df_nan(library="pandas"):
    return dataframe_init_dispatch(
        dataframe_dict={"a": [None, None, None]},
        library=library,
    )


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values(self, library):
        """Test that the impute values learnt during fit are expected."""

        df_dict = {
            # skipping int, as non-nullable for pandas
            # float
            "float_col": [1.0, 2.0, 1.0, None, 3.0],
            # str
            "str_col": ["a", "b", "b", None, None],
            # bool
            "bool_col": [True, True, None, False, True],
            # cat
            "cat_col": ["b", "b", "d", "e", None],
        }

        df = dataframe_init_dispatch(df_dict, library=library)

        # create categorical col
        df = nw.from_native(df)
        df = nw.to_native(df.with_columns(nw.col("cat_col").cast(nw.Categorical)))

        x = ModeImputer(columns=["float_col", "str_col", "cat_col", "bool_col"])

        x.fit(df)

        expected_impute_values = {
            "float_col": 1.0,
            "str_col": "b",
            "cat_col": "b",
            "bool_col": True,
        }

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values_tied(self, library):
        """Test that the impute values learnt during fit are expected - when mode is tied."""

        df_dict = {
            # skipping int, as non-nullable for pandas
            # float
            "float_col": [1.0, 2.0, 1.0, None, 2.0],
            # str
            "str_col": ["a", "b", "b", "a", None],
            # bool
            "bool_col": [True, True, None, False, False],
            # cat
            "cat_col": ["b", "b", "e", "e", None],
        }

        df = dataframe_init_dispatch(df_dict, library=library)

        # create categorical col
        df = nw.from_native(df)
        df = nw.to_native(df.with_columns(nw.col("cat_col").cast(nw.Categorical)))

        columns = ["float_col", "str_col", "cat_col", "bool_col"]
        x = ModeImputer(columns=columns)

        x.fit(df)

        expected_impute_values = {
            "float_col": 2.0,
            "str_col": "b",
            "cat_col": "e",
            "bool_col": True,
        }

        with pytest.warns(
            UserWarning,
        ) as warnings:
            x.fit(df)

        columns.sort()
        warnings = [w.message.args[0] for w in warnings]
        for col in columns:
            msg = f"ModeImputer: The Mode of column {col} is tied, will sort in descending order and return first candidate"
            assert msg in warnings

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    @pytest.mark.parametrize(
        ("input_col", "weight_col", "learnt_value", "categorical"),
        [
            # float
            ([1.0, 2.0, None], [2, 2, 2], 2.0, False),
            # str
            (["a", "b", "a", "b", None], [2, 1, 1, 2, 4], "b", False),
            # bool
            ([True, False, None], [4, 4, 1], True, False),
            # cat
            (["g", "g", "h", None], [2, 2, 4, 1], "h", True),
        ],
    )
    def test_learnt_values_tied_weighted(
        self,
        library,
        input_col,
        weight_col,
        learnt_value,
        categorical,
    ):
        """
        Test that the impute values learnt during fit are expected -
        when mode is tied and transformer is weighted.
        """
        df_dict = {
            "col": input_col,
            "weight": weight_col,
        }

        df = dataframe_init_dispatch(df_dict, library=library)

        if categorical:
            # create categorical col
            df = nw.from_native(df)
            df = nw.to_native(df.with_columns(nw.col("col").cast(nw.Categorical)))

        columns = ["col"]
        x = ModeImputer(columns=columns, weights_column="weight")

        x.fit(df)

        expected_impute_values = {
            "col": learnt_value,
        }
        with pytest.warns(
            UserWarning,
        ) as warnings:
            x.fit(df)

        columns.sort()
        warnings = [w.message.args[0] for w in warnings]
        for col in columns:
            msg = f"ModeImputer: The Mode of column {col} is tied, will sort in descending order and return first candidate"
            assert msg in warnings

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nan_learnt_values(self, library):
        """Test behaviour when learnt value is None."""
        x = ModeImputer(columns=["a"])

        df = input_df_nan(library)

        with pytest.warns(
            UserWarning,
            match=r"ModeImputer: The Mode of columns \['a'\] will be None",
        ):
            x.fit(df)

        expected_impute_values = {"a": None}

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nan_learnt_values_weighted(self, library):
        """Test behaviour when learnt value is None - when weights are used."""
        weights_column = "weights_column"
        x = ModeImputer(columns=["a"], weights_column=weights_column)

        df = d.create_weighted_imputers_test_df(library=library)

        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)

        # replace 'a' with all null values to trigger warning
        df = df.with_columns(
            nw.new_series(
                name="a",
                values=[None] * len(df),
                backend=native_backend,
            ),
        )

        df = df.to_native()

        with pytest.warns(
            UserWarning,
            match=r"ModeImputer: The Mode of columns \['a'\] will be None",
        ):
            x.fit(df)

        expected_impute_values = {"a": None}

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    @pytest.mark.parametrize(
        ("input_col", "weight_col", "learnt_value", "categorical"),
        [
            # float
            ([1.0, 2.0, 2.0, np.nan], [2, 2, 2, 1], 2.0, False),
            # str
            (["a", "b", "a", "b", None, "b"], [2, 1, 1, 2, 4, 3], "b", False),
            # bool
            ([True, False, None, True, True], [4, 4, 1, 1, 1], True, False),
            # cat
            (["a", "b", "c", "c", None], [1, 2, 3, 4, 5], "c", True),
        ],
    )
    def test_learnt_values_weighted_df(
        self,
        library,
        input_col,
        weight_col,
        learnt_value,
        categorical,
    ):
        """Test that the impute values learnt during fit are expected when df is weighted."""
        df_dict = {
            "col": input_col,
            "weight": weight_col,
        }

        df = dataframe_init_dispatch(df_dict, library=library)

        if categorical:
            # create categorical col
            df = nw.from_native(df)
            df = nw.to_native(df.with_columns(nw.col("col").cast(nw.Categorical)))

        columns = ["col"]
        x = ModeImputer(columns=columns, weights_column="weight")

        x.fit(df)

        expected_impute_values = {
            "col": learnt_value,
        }

        assert x.impute_values_ == expected_impute_values, (
            f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"
        )


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTestsWeight,
    GenericImputerTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"
