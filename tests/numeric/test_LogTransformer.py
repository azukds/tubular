import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import DropOriginalInitMixinTests, DropOriginalTransformMixinTests
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import LogTransformer


class TestInit(BaseNumericTransformerInitTests, DropOriginalInitMixinTests):
    """Tests for LogTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "LogTransformer"

    def test_base_type_error(self):
        """Test that an exception is raised if base is non-numeric."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            LogTransformer(
                columns=["a"],
                base="a",
            )

    def test_suffix_type_error(self):
        """Test that an exception is raised if suffix is non-str."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            LogTransformer(
                columns=["a"],
                suffix=1,
            )

    def test_add_1_type_error(self):
        """Test that an exception is raised if add_1 is not bool."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            LogTransformer(
                columns=["a"],
                add_1="bla",
            )

    def test_base_not_strictly_positive_error(self):
        """Test that an exception is raised if base is not strictly positive."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            LogTransformer(
                columns=["a"],
                base=0,
            )


class TestTransform(
    BaseNumericTransformerTransformTests,
    DropOriginalTransformMixinTests,
):
    """Tests for LogTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "LogTransformer"

    def expected_df_1():
        """Expected output of test_expected_output_1."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"])
        df["b_new_col"] = np.log(df["b"])

        return df.drop(columns=["a", "b"])

    def expected_df_2():
        """Expected output of test_expected_output_2."""
        df = d.create_df_3()

        df["a_new_col"] = np.log1p(df["a"])
        df["b_new_col"] = np.log1p(df["b"])

        return df.drop(columns=["a", "b"])

    def expected_df_3():
        """Expected output of test_expected_output_3."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"])
        df["b_new_col"] = np.log(df["b"])

        return df

    def expected_df_4():
        """Expected output of test_expected_output_4."""
        df = d.create_df_3()

        df["a_new_col"] = np.log1p(df["a"])
        df["b_new_col"] = np.log1p(df["b"])

        return df

    def expected_df_5():
        """Expected output of test_expected_output_5."""
        df = d.create_df_4()

        df["a_new_col"] = np.log(df["a"] + 1) / np.log(5)

        return df

    def expected_df_6():
        """Expected output of test_expected_output_6."""
        df = d.create_df_4()

        df["a_new_col"] = np.log(df["a"]) / np.log(7)

        return df.drop("a", axis=1)

    def test_log1p(self):
        """Test that log1p is working as intended."""
        df = pd.DataFrame(
            {
                "a": [0.00001, 0.00002, 0.00003],
                "b": [0.00004, 0.00005, 0.00006],
            },
        )
        # Values created using np.log1p() of original df
        expected = pd.DataFrame(
            {
                "a_log": [9.999950e-06, 1.999980e-05, 2.999955e-05],
                "b_log": [3.99992000e-05, 4.99987500e-05, 5.99982001e-05],
            },
        )
        log_transformer = LogTransformer(
            columns=["a", "b"],
            add_1=True,
        )
        actual = log_transformer.transform(df)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_1(self, df, expected):
        """Test that transform is giving the expected output when not adding one and dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"],
            add_1=False,
            drop_original=True,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_2(self, df, expected):
        """Test that transform is giving the expected output when adding one and dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"],
            add_1=True,
            drop_original=True,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_3()),
    )
    def test_expected_output_3(self, df, expected):
        """Test that transform is giving the expected output when not adding one and not dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"],
            add_1=False,
            drop_original=False,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_4()),
    )
    def test_expected_output_4(self, df, expected):
        """Test that transform is giving the expected output when adding one and not dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"],
            add_1=True,
            drop_original=False,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_4(), expected_df_5()),
    )
    def test_expected_output_5(self, df, expected):
        """Test that transform is giving the expected output when adding one and not dropping
        original columns and using base.
        """
        x1 = LogTransformer(
            columns=["a"],
            base=5,
            add_1=True,
            drop_original=False,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_4(), expected_df_6()),
    )
    def test_expected_output_6(self, df, expected):
        """Test that transform is giving the expected output when  not adding one and dropping
        original columns and using base.
        """
        x1 = LogTransformer(
            columns=["a"],
            base=7,
            add_1=False,
            drop_original=True,
            suffix="new_col",
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform should be using base, not adding 1, and not dropping original columns",
        )

    @pytest.mark.parametrize(
        ("df", "columns", "add_1", "extra_exception_text"),
        (
            [pd.DataFrame({"a": [1, 2, 0]}), ["a"], False, ""],
            [pd.DataFrame({"a": [1, 2, 0], "b": [1, 2, 3]}), ["a", "b"], False, ""],
            [pd.DataFrame({"a": [1, 2, -1]}), ["a"], True, r" \(after adding 1\)"],
            [
                pd.DataFrame({"a": [1, 2, -1], "b": [1, 2, 3]}),
                ["a", "b"],
                True,
                r" \(after adding 1\)",
            ],
            [pd.DataFrame({"b": [1, 2, -0.001]}), ["b"], False, ""],
            [
                pd.DataFrame({"b": [1, 2, -0.001], "a": [1, 2, 3]}),
                ["a", "b"],
                False,
                "",
            ],
            [pd.DataFrame({"b": [1, 2, -1.001]}), ["b"], True, r" \(after adding 1\)"],
            [
                pd.DataFrame({"b": [1, 2, -1.001], "a": [1, 2, 3]}),
                ["a", "b"],
                True,
                r" \(after adding 1\)",
            ],
        ),
    )
    def test_negative_values_raise_exception(
        self,
        df,
        columns,
        add_1,
        extra_exception_text,
    ):
        """Test that an exception is raised if negative values are passed in transform."""
        x = LogTransformer(columns=columns, add_1=add_1, drop_original=True)

        with pytest.raises(
            ValueError,
            match=f"LogTransformer: values less than or equal to 0 in columns{extra_exception_text}, make greater than 0 before using transform",
        ):
            x.transform(df)
