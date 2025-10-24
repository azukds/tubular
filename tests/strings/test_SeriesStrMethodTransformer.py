import numpy as np
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tubular.strings import SeriesStrMethodTransformer


class TestInit(ColumnStrListInitTests, NewColumnNameInitMixintests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesStrMethodTransformer"

    # Duplicated here as base test sets "columns" to a list of len(2), not viable for this transformer
    @pytest.mark.parametrize(
        "non_string",
        [1, True, {"a": 1}, [1, 2], None, np.inf, np.nan],
    )
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = [non_string]

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)

    def test_list_length(self):
        """Test that an error is raised if columns list contains more than a single element"""

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name=1,
                columns=["b", "c"],
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test and exception is raised if a non pd.Series.str method is passed for pd_method_name."""
        with pytest.raises(
            AttributeError,
            match=r"""SeriesStrMethodTransformer: error accessing "str.b" method on pd.Series object - pd_method_name should be a pd.Series.str method""",
        ):
            SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name="b",
                columns=["b"],
            )

    @pytest.mark.parametrize(
        "non_dict",
        [1, "a", True, [1, 2], np.inf, np.nan],
    )
    def test_invalid_pd_kwargs_type_errors(
        self,
        non_dict,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exceptions are raised for invalid pd_kwargs types."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["pd_method_kwargs"] = non_dict

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize(
        "na_dict_key",
        [{"a": 1, 2: "b"}, {"a": 1, (1, 2): "b"}],
    )
    def test_invalid_pd_kwargs_key_errors(
        self,
        na_dict_key,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exceptions are raised for invalid pd_kwargs key types."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["pd_method_kwargs"] = na_dict_key

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestTransform(GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesStrMethodTransformer"

    def expected_df_1():
        """Expected output of test_expected_output_no_overwrite."""
        df = d.create_df_7()

        df["b_new"] = df["b"].str.find(sub="a")

        return df

    def expected_df_2():
        """Expected output of test_expected_output_overwrite."""
        df = d.create_df_7()

        df["b"] = df["b"].str.pad(width=10)

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_7(), expected_df_1()),
    )
    def test_expected_output_no_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when not overwriting the original column."""
        x = SeriesStrMethodTransformer(
            new_column_name="b_new",
            pd_method_name="find",
            columns=["b"],
            pd_method_kwargs={"sub": "a"},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesStrMethodTransformer.transform with find, not overwriting original column",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_7(), expected_df_2()),
    )
    def test_expected_output_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when overwriting the original column."""
        x = SeriesStrMethodTransformer(
            new_column_name="b",
            pd_method_name="pad",
            columns=["b"],
            pd_method_kwargs={"width": 10},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesStrMethodTransformer.transform with pad, overwriting original column",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SeriesStrMethodTransformer"
