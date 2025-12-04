import numpy as np
import pandas as pd
import pytest
import test_aide as ta

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tubular.misc import ColumnDtypeSetter


class TestInit(ColumnStrListInitTests):
    """Generic tests for ColumnDtypeSetter.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"

    @pytest.mark.parametrize(
        "invalid_dtype",
        ["STRING", "misc_invalid", "np.int", 0],
    )
    def test_invalid_dtype_error(self, invalid_dtype):
        msg = f"ColumnDtypeSetter: data type '{invalid_dtype}' not understood as a valid dtype"
        with pytest.raises(TypeError, match=msg):
            ColumnDtypeSetter(columns=["a"], dtype=invalid_dtype)


class TestFit(GenericFitTests):
    """Generic tests for ColumnDtypeSetter.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"


class TestTransform(GenericTransformTests):
    """Tests for ColumnDtypeSetter.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"

    def base_df():
        """Input dataframe from test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
                "b": [1.0, 2.0, 3.0, np.nan, 7.0, 8.0, 9.0],
                "c": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
                "d": [1, 1, 2, 3, -4, -5, -6],
            },
        )

    def expected_df():
        """Expected output from test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
                "b": [1.0, 2.0, 3.0, np.nan, 7.0, 8.0, 9.0],
                "c": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
                "d": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(base_df(), expected_df())
        + ta.pandas.index_preserved_params(base_df(), expected_df()),
    )
    @pytest.mark.parametrize("dtype", [float, "float"])
    def test_expected_output(self, df, expected, dtype):
        """Test values are correctly set to float dtype."""
        df["a"] = df["a"].astype(str)
        df["b"] = df["b"].astype(float)
        df["c"] = df["c"].astype(int)
        df["d"] = df["d"].astype(str)

        x = ColumnDtypeSetter(columns=["a", "b", "c", "d"], dtype=dtype)

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check values correctly converted to float",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for ColumnDtypeSetter behaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"
