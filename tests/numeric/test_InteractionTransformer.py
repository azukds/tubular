import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import InteractionTransformer


class TestInit(BaseNumericTransformerInitTests):
    """Tests for InteractionTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "InteractionTransformer"

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree="2",
                max_degree=2,
            )
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=2,
                max_degree="2",
            )

    def test_invalid_input_value_errors(self):
        """Test and exception is raised if degrees or columns provided are inconsistent."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            InteractionTransformer(
                columns=["A"],
            )

        with pytest.raises(
            ValueError,
            match=r"""InteractionTransformer: min_degree must be equal or greater than 2, got 0""",
        ):
            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=0,
                max_degree=2,
            )

        with pytest.raises(
            ValueError,
            match=r"""InteractionTransformer: max_degree must be equal or greater than min_degree""",
        ):
            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=3,
                max_degree=2,
            )
        # NEW
        with pytest.raises(
            ValueError,
            match=r"""InteractionTransformer: max_degree must be equal or lower than number of columns""",
        ):
            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=3,
                max_degree=4,
            )


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for InteractionTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "InteractionTransformer"

    def expected_df_1():
        """Expected output of test_expected_output_default_assignment."""
        return pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.nan},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.nan, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.nan, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "a b": {0: 1.0, 1: 4.0, 2: 9.0, 3: np.nan, 4: 35.0, 5: 48.0, 6: np.nan},
                "a c": {
                    0: np.nan,
                    1: 2.0,
                    2: 6.0,
                    3: 12.0,
                    4: -20.0,
                    5: -30.0,
                    6: np.nan,
                },
                "b c": {
                    0: np.nan,
                    1: 2.0,
                    2: 6.0,
                    3: np.nan,
                    4: -28.0,
                    5: -40.0,
                    6: -54.0,
                },
            },
        )

    def expected_df_2():
        """Expected output of test_expected_output_multiple_columns_assignment."""
        return pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.nan},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.nan, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.nan, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "a b": {0: 1.0, 1: 4.0, 2: 9.0, 3: np.nan, 4: 35.0, 5: 48.0, 6: np.nan},
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_default_assignment(self, df, expected):
        """Test default values and multiple columns assignment from transform gives expected results."""
        x = InteractionTransformer(columns=["a", "b", "c"])

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer default values",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_multiple_columns_assignment(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""
        x = InteractionTransformer(columns=["a", "b"])

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )
