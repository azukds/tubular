import numpy as np
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.misc import SetValueTransformer


def expected_df_1(library, value):
    """Expected output of test_value_set_in_transform."""

    df_dict = {
        "a": [value] * 7,
        "b": [value] * 7,
        "c": ["a", "b", "c", "d", "e", "f", None],
    }

    df = dataframe_init_dispatch(df_dict, library)
    if library == "pandas":
        df["c"] = df["c"].astype("category")
    elif library == "polars":
        df = df.with_columns(df["c"].cast(pl.Categorical))
        # polars automatically downcasts to int32 in transformer
        if isinstance(value, int):
            df = df.with_columns(df[["a", "b"]].cast(pl.Int32))
    return df


class TestInit(ColumnStrListInitTests):
    """Generic tests for SetValueTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"


class TestFit(GenericFitTests):
    """Generic tests for SetValueTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"


class TestTransform(GenericTransformTests):
    """Tests for SetValueTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("value", ["a", 1, 1.0, None, np.nan])
    def test_value_set_in_transform(self, library, value, lazy):
        """Test that transform sets the value as expected."""

        df = d.create_df_2(library)

        x = SetValueTransformer(columns=["a", "b"], value=value)

        polars = isinstance(df, pl.DataFrame)

        if _check_if_skip_test(x, df, lazy):
            return

        df_transformed = x.transform(_convert_to_lazy(df, lazy))

        expected = expected_df_1(library, value)

        assert_frame_equal_dispatch(
            df1=_collect_frame(df_transformed, polars, lazy),
            df2=expected,
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for SetValueTransformer behaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"
