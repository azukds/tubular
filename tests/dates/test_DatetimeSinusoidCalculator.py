import narwhals as nw
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseDatetimeTransformer import DatetimeMixinTransformTests
from tests.utils import assert_frame_equal_dispatch
from tubular.dates import DatetimeSinusoidCalculator


@pytest.fixture(scope="module", autouse=True)
def example_transformer():
    return DatetimeSinusoidCalculator("a", "cos", "hour", 24)


class TestInit(
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
):
    """Tests for DatetimeSinusoidCalculator.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeSinusoidCalculator"

    @pytest.mark.parametrize("incorrect_type_method", [2, 2.0, True, {"a": 4}])
    def test_method_type_error(self, incorrect_type_method):
        """Test that an exception is raised if method is not a str or a list."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                incorrect_type_method,
                "hour",
                24,
            )

    @pytest.mark.parametrize("incorrect_type_units", [2, 2.0, True, ["help"]])
    def test_units_type_error(self, incorrect_type_units):
        """Test that an exception is raised if units is not a str or a dict."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_type_units,
                24,
            )

    @pytest.mark.parametrize("incorrect_type_period", ["2", True, ["help"]])
    def test_period_type_error(self, incorrect_type_period):
        """Test that an error is raised if period is not an int or a float or a dictionary."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "hour",
                incorrect_type_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_types_period",
        [{"str": True}, {2: "str"}, {2: 2}, {"str": ["str"]}],
    )
    def test_period_dict_type_error(self, incorrect_dict_types_period):
        """Test that an error is raised if period dict is not a str:int or str:float kv pair."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "hour",
                incorrect_dict_types_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_types_units",
        [
            {"str": True},
            {2: "str"},
            {"str": 2},
            {2: 2},
            {"str": True},
            {"str": ["str"]},
        ],
    )
    def test_units_dict_type_error(self, incorrect_dict_types_units):
        """Test that an error is raised if units dict is not a str:str kv pair."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_dict_types_units,
                24,
            )

    @pytest.mark.parametrize("incorrect_dict_units", [{"str": "tweet"}])
    def test_units_dict_value_error(self, incorrect_dict_units):
        """Test that an error is raised if units dict value is not from the valid units list."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_dict_units,
                24,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_columns_period",
        [{"ham": 24}, {"str": 34.0}],
    )
    def test_period_dict_col_error(self, incorrect_dict_columns_period):
        """Test that an error is raised if period dict keys are not equal to columns."""
        with pytest.raises(
            ValueError,
            match=f"period dictionary keys must be the same as columns but got {set(incorrect_dict_columns_period.keys())}",
        ):
            DatetimeSinusoidCalculator(
                ["vegan_sausages", "carrots", "peas"],
                "cos",
                "hour",
                incorrect_dict_columns_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_columns_unit",
        [{"sausage_roll": "hour"}],
    )
    def test_unit_dict_col_error(self, incorrect_dict_columns_unit):
        """Test that an error is raised if unit dict keys is not equal to columns."""
        with pytest.raises(
            ValueError,
            match=f"unit dictionary keys must be the same as columns but got {set(incorrect_dict_columns_unit.keys())}",
        ):
            DatetimeSinusoidCalculator(
                ["vegan_sausages", "carrots", "peas"],
                "cos",
                incorrect_dict_columns_unit,
                6,
            )

    def test_valid_method_value_error(self):
        """Test that a value error is raised if method is not sin, cos or a list containing both."""
        method = "tan"

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                method,
                "year",
                24,
            )

    def test_valid_units_value_error(self):
        """Test that a value error is raised if the unit supplied is not in the valid units list."""
        units = "five"

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                units,
                24,
            )


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeSinusoidCalculator"


class TestTransform(GenericTransformTests, DatetimeMixinTransformTests):
    """Tests for BaseTwoColumnDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeSinusoidCalculator"

    @pytest.mark.parametrize(
        "columns, method, units, period",
        [
            (["a", "b"], "cos", "month", 12),
            (["a"], "cos", "month", 12),
        ],
    )
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_single_method(
        self,
        columns,
        method,
        units,
        period,
        library,
    ):
        """Test that the transformer produces the expected output for a single method."""
        expected_df = nw.from_native(d.create_datediff_test_df(library=library))
        transformer = DatetimeSinusoidCalculator(
            columns=columns,
            method=method,
            units=units,
            period=period,
        )

        expected = expected_df.clone()
        native_backend = nw.get_native_namespace(expected)

        for column in transformer.columns:
            new_col_name = f"cos_12_month_{column}"
            # Get the column in the desired unit
            sinusoid_col = np.cos(expected[column].dt.month() * (2.0 * np.pi / 12))
            expected = expected.with_columns(
                nw.new_series(
                    name=new_col_name,
                    values=sinusoid_col,
                    backend=native_backend.__name__,
                ),
            )

        df = nw.from_native(d.create_datediff_test_df(library=library))
        actual = transformer.transform(df)
        assert_frame_equal_dispatch(
            actual,
            expected.to_native(),
        )
        # also check single rows
        for i in range(len(df)):
            actual_row = transformer.transform(df[[i]].to_native())
            expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                actual_row,
                expected_row,
            )

    @pytest.mark.parametrize(
        "columns, method, units, period",
        [
            (["a"], ["sin", "cos"], "month", 12),
        ],
    )
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_both_methods_single_column(
        self,
        columns,
        method,
        units,
        period,
        library,
    ):
        """Test that the transformer produces the expected output for both methods on a single column."""
        expected_df = nw.from_native(d.create_datediff_test_df(library=library))
        transformer = DatetimeSinusoidCalculator(
            method=method,
            units=units,
            period=period,
            columns=columns,
        )
        expected = expected_df.clone()
        native_backend = nw.get_native_namespace(expected)

        # Create new column names for sin and cos
        new_cos_col_name = "cos_12_month_a"
        new_sin_col_name = "sin_12_month_a"
        # Get the column in the desired unit
        column_in_desired_unit = expected["a"].dt.month

        expected = expected.with_columns(
            nw.new_series(
                name=new_sin_col_name,
                values=np.sin(column_in_desired_unit() * (2.0 * np.pi / 12)),
                backend=native_backend.__name__,
            ),
            nw.new_series(
                name=new_cos_col_name,
                values=np.cos(column_in_desired_unit() * (2.0 * np.pi / 12)),
                backend=native_backend.__name__,
            ),
        )

        df = nw.from_native(d.create_datediff_test_df(library=library))
        actual = transformer.transform(df)
        assert_frame_equal_dispatch(
            actual,
            expected.to_native(),
        )

        # also check single rows
        for i in range(len(df)):
            actual_row = transformer.transform(df[[i]].to_native())
            expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                actual_row,
                expected_row,
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_dict_units(self, library):
        """Test that the transformer produces the expected output when units is a dictionary."""
        expected_df = nw.from_native(d.create_datediff_test_df(library=library))
        transformer = DatetimeSinusoidCalculator(
            columns=["a", "b"],
            method=["sin"],
            units={"a": "month", "b": "day"},
            period=12,
        )

        expected = expected_df.clone()
        native_backend = nw.get_native_namespace(expected)
        # Calculate the sine values for the month of column 'a' and day of column 'b'
        sin_12_month_a = np.sin(expected["a"].dt.month() * (2.0 * np.pi / 12))
        sin_12_day_b = np.sin(expected["b"].dt.day() * (2.0 * np.pi / 12))
        # Add the new columns to the expected DataFrame
        expected = expected.with_columns(
            nw.new_series(
                name="sin_12_month_a",
                values=sin_12_month_a,
                backend=native_backend.__name__,
            ),
            nw.new_series(
                name="sin_12_day_b",
                values=sin_12_day_b,
                backend=native_backend.__name__,
            ),
        )

        df = nw.from_native(d.create_datediff_test_df(library=library))

        actual = transformer.transform(df)
        assert_frame_equal_dispatch(
            actual,
            expected.to_native(),
        )
        # also check single rows
        for i in range(len(df)):
            actual_row = transformer.transform(df[[i]].to_native())
            expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                actual_row,
                expected_row,
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_dict_period(self, library):
        """Test that the transformer produces the expected output when period is a dictionary."""
        expected_df = nw.from_native(d.create_datediff_test_df(library=library))
        transformer = DatetimeSinusoidCalculator(
            columns=["a", "b"],
            method=["sin"],
            units="month",
            period={"a": 12, "b": 24},
        )

        expected = expected_df.clone()
        native_backend = nw.get_native_namespace(expected)
        # Calculate the sine values for the month of column 'a' and 'b'
        sin_12_month_a = np.sin(expected["a"].dt.month() * (2.0 * np.pi / 12))
        sin_24_month_b = np.sin(expected["b"].dt.month() * (2.0 * np.pi / 24))
        # Add the new columns to the expected DataFrame
        expected = expected.with_columns(
            nw.new_series(
                name="sin_12_month_a",
                values=sin_12_month_a,
                backend=native_backend.__name__,
            ),
            nw.new_series(
                name="sin_24_month_b",
                values=sin_24_month_b,
                backend=native_backend.__name__,
            ),
        )

        df = nw.from_native(d.create_datediff_test_df(library=library))
        actual = transformer.transform(df)
        assert_frame_equal_dispatch(
            actual,
            expected.to_native(),
        )
        # also check single rows
        for i in range(len(df)):
            actual_row = transformer.transform(df[[i]].to_native())
            expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                actual_row,
                expected_row,
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_dict_units_and_period(self, library):
        """Test that the transformer produces the expected output when both units and period are dictionaries."""
        expected_df = nw.from_native(d.create_datediff_test_df(library=library))
        transformer = DatetimeSinusoidCalculator(
            columns=["a", "b"],
            method=["sin"],
            units={"a": "month", "b": "day"},
            period={"a": 12, "b": 24},
        )

        expected = expected_df.clone()
        native_backend = nw.get_native_namespace(expected)
        # Calculate the sine values for the month of column 'a' and day of column 'b'
        sin_12_month_a = np.sin(expected["a"].dt.month() * (2.0 * np.pi / 12))
        sin_24_day_b = np.sin(expected["b"].dt.day() * (2.0 * np.pi / 24))
        # Add the new columns to the expected DataFrame
        expected = expected.with_columns(
            nw.new_series(
                name="sin_12_month_a",
                values=sin_12_month_a,
                backend=native_backend.__name__,
            ),
            nw.new_series(
                name="sin_24_day_b",
                values=sin_24_day_b,
                backend=native_backend.__name__,
            ),
        )

        df = nw.from_native(d.create_datediff_test_df(library=library))
        actual = transformer.transform(df)
        assert_frame_equal_dispatch(
            actual,
            expected.to_native(),
        )
        # also check single rows
        for i in range(len(df)):
            actual_row = transformer.transform(df[[i]].to_native())
            expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                actual_row,
                expected_row,
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeSinusoidCalculator"
