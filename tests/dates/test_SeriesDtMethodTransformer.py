import numpy as np
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseDatetimeTransformer import (
    DatetimeMixinTransformTests,
)
from tubular.dates import SeriesDtMethodTransformer


class TestInit(
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    NewColumnNameInitMixintests,
):
    """Tests for SeriesDtMethodTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesDtMethodTransformer"

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""
        bad_columns = ["b", "c"]
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name=1,
                columns=bad_columns,
            )

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name=1,
                columns="b",
            )

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                columns="b",
                pd_method_kwargs=1,
            )

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                columns="b",
                pd_method_kwargs={"a": 1, 2: "b"},
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test and exception is raised if a non pd.Series.dt method is passed for pd_method_name."""
        with pytest.raises(
            AttributeError,
            match=r"""SeriesDtMethodTransformer: error accessing "dt.b" method on pd.Series object - pd_method_name should be a pd.Series.dt method""",
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="b",
                columns="b",
            )


class TestTransform(
    DatetimeMixinTransformTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
):
    """Tests for SeriesDtMethodTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesDtMethodTransformer"

    def expected_df_1():
        """Expected output of test_expected_output_no_overwrite."""
        df = d.create_datediff_test_df()

        df["a_year"] = np.array(
            [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985],
            dtype=np.int32,
        )

        return df

    def expected_df_2():
        """Expected output of test_expected_output_overwrite."""
        df = d.create_datediff_test_df()

        df["a"] = np.array(
            [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985],
            dtype=np.int32,
        )

        return df

    def expected_df_3():
        """Expected output of test_expected_output_callable."""
        df = d.create_datediff_test_df()

        df["b_new"] = df["b"].dt.to_period("M")

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_1(),
        ),
    )
    def test_expected_output_no_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when not overwriting the original column."""
        x = SeriesDtMethodTransformer(
            new_column_name="a_year",
            pd_method_name="year",
            columns="a",
            pd_method_kwargs=None,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with find, not overwriting original column",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_2(),
        ),
    )
    def test_expected_output_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when overwriting the original column."""
        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name="year",
            columns="a",
            pd_method_kwargs=None,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with pad, overwriting original column",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_3(),
        ),
    )
    def test_expected_output_callable(self, df, expected):
        """Test transform gives expected results, when pd_method_name is a callable."""
        x = SeriesDtMethodTransformer(
            new_column_name="b_new",
            pd_method_name="to_period",
            columns="b",
            pd_method_kwargs={"freq": "M"},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with to_period",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesDtMethodTransformer"
