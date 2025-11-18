import numpy as np
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _handle_from_json,
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

    @pytest.mark.parametrize("value", [{"a": 1}, [1, 2]])
    def test_value_arg_type(self, value):
        """Tests that check arg value type."""

        with pytest.raises(BeartypeCallHintParamViolation):
            SetValueTransformer(columns=["a"], value=value)


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

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("value", ["a", 1, 1.0, None, np.nan])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_value_set_in_transform(self, library, value, from_json):
        """Test that transform sets the value as expected."""

        df = d.create_df_2(library)

        x = SetValueTransformer(columns=["a", "b"], value=value)

        if from_json:
            x = _handle_from_json(x, from_json)

        df_transformed = x.transform(df)

        expected = expected_df_1(library, value)

        assert_frame_equal_dispatch(
            df1=df_transformed,
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

    @pytest.mark.parametrize("value", ["a", 1, 1.0, None, np.nan])
    def test_to_json_returns_correct_dict(self, value):
        """Test that to_json is working as expected."""
        transformer = SetValueTransformer(columns="a", value=value)

        actual = transformer.to_json()

        # check tubular_version is present and a string, then remove
        assert isinstance(
            actual["tubular_version"],
            str,
        ), "expected tubular version to be captured as str in to_json"
        del actual["tubular_version"]

        expected = {
            "classname": "SetValueTransformer",
            "init": {
                "columns": ["a"],
                "copy": False,
                "verbose": False,
                "return_native": True,
                "value": value,
            },
            "fit": {},
        }

        assert actual == expected, "to_json does not return the expected dictionary"
