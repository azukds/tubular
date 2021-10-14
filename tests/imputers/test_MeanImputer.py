import pytest
import test_aide as ta
import pandas as pd
import numpy as np

import tubular
from tubular.imputers import MeanImputer


class TestInit(object):
    """Tests for MeanImputer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=MeanImputer.__init__,
            expected_arguments=["self", "columns"],
            expected_default_values=(None,),
        )

    def test_class_methods(self):
        """Test that MeanImputer has fit and transform methods."""

        x = MeanImputer()

        ta.class_helpers.test_object_method(obj=x, expected_method="fit", msg="fit")

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that MeanImputer inherits from BaseImputer."""

        x = MeanImputer()

        ta.class_helpers.assert_inheritance(x, tubular.imputers.BaseImputer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": None, "verbose": True, "copy": True}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            MeanImputer(columns=None, verbose=True, copy=True)


class TestFit(object):
    """Tests for MeanImputer.fit()"""

    def test_arguments(self):
        """Test that init fit expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=MeanImputer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_called(self, mocker):
        """Test that fit calls BaseTransformer.fit."""

        df = ta.test_data.create_df_3()

        x = MeanImputer(columns=["a", "b", "c"])

        expected_call_args = {
            0: {"args": (ta.test_data.create_df_3(), None), "kwargs": {}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "fit", expected_call_args
        ):

            x.fit(df)

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""

        df = ta.test_data.create_df_3()

        x = MeanImputer(columns=["a", "b", "c"])

        x.fit(df)

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": df["a"].mean(),
                    "b": df["b"].mean(),
                    "c": df["c"].mean(),
                }
            },
            msg="impute_values_ attribute",
        )

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = ta.test_data.create_df_1()

        x = MeanImputer(columns="a")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from MeanImputer.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""

        df = ta.test_data.create_df_1()

        x = MeanImputer(columns="a")

        x.fit(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=ta.test_data.create_df_1(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform(object):
    """Tests for MeanImputer.transform()."""

    def expected_df_1():
        """Expected output for test_nulls_imputed_correctly."""

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
            }
        )

        for col in ["a", "b", "c"]:

            df[col].loc[df[col].isnull()] = df[col].mean()

        return df

    def expected_df_2():
        """Expected output for test_nulls_imputed_correctly_2."""

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
            }
        )

        for col in ["a"]:

            df[col].loc[df[col].isnull()] = df[col].mean()

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=MeanImputer.transform, expected_arguments=["self", "X"]
        )

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""

        df = ta.test_data.create_df_1()

        x = MeanImputer(columns="a")

        x.fit(df)

        expected_call_args = {0: {"args": (["impute_values_"],), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = ta.test_data.create_df_1()

        x = MeanImputer(columns="a")

        x.fit(df)

        expected_call_args = {0: {"args": (ta.test_data.create_df_1(),), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(ta.test_data.create_df_3(), expected_df_1())
        + ta.pandas_helpers.index_preserved_params(
            ta.test_data.create_df_3(), expected_df_1()
        ),
    )
    def test_nulls_imputed_correctly(self, df, expected):
        """Test missing values are filled with the correct values."""

        x = MeanImputer(columns=["a", "b", "c"])

        # set the impute values dict directly rather than fitting x on df so test works with decorators
        x.impute_values_ = {"a": 3.5, "b": 5.0, "c": -1.5}

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(ta.test_data.create_df_3(), expected_df_2())
        + ta.pandas_helpers.index_preserved_params(
            ta.test_data.create_df_3(), expected_df_2()
        ),
    )
    def test_nulls_imputed_correctly_2(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are not changed."""

        x = MeanImputer(columns=["a"])

        # set the impute values dict directly rather than fitting x on df so test works with decorators
        x.impute_values_ = {"a": 3.5}

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    def test_learnt_values_not_modified(self):
        """Test that the impute_values_ from fit are not changed in transform."""

        df = ta.test_data.create_df_3()

        x = MeanImputer(columns=["a", "b", "c"])

        x.fit(df)

        x2 = MeanImputer(columns=["a", "b", "c"])

        x2.fit_transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )
