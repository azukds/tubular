import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import TwoColumnOperatorTransformer


class TestInit(
    NewColumnNameInitMixintests,
    TwoColumnListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "TwoColumnOperatorTransformer"

    def test_axis_not_present_error(self):
        """Checks that an error is raised if no axis element present in pd_method_kwargs dict."""
        with pytest.raises(
            ValueError,
            match='pd_method_kwargs must contain an entry "axis" set to 0 or 1',
        ):
            TwoColumnOperatorTransformer("mul", ["a", "b"], "c", pd_method_kwargs={})

    def test_axis_not_valid_error(self):
        """Checks that an error is raised if no axis element present in pd_method_kwargs dict."""
        with pytest.raises(ValueError, match="pd_method_kwargs 'axis' must be 0 or 1"):
            TwoColumnOperatorTransformer(
                "mul",
                ["a", "b"],
                "c",
                pd_method_kwargs={"axis": 2},
            )


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "TwoColumnOperatorTransformer"

    @pytest.mark.parametrize(
        ("pd_method_name", "output"),
        [
            (
                "mul",
                [4, 10, 18],
            ),
            ("div", [0.25, 0.4, 0.5]),
            ("pow", [1, 32, 729]),
        ],
    )
    def test_expected_output(self, pd_method_name, output):
        """Tests that the output given by TwoColumnOperatorTransformer is as you would expect."""
        expected = d.create_df_11()
        expected["c"] = output
        x = TwoColumnOperatorTransformer(
            pd_method_name,
            ["a", "b"],
            "c",
        )
        actual = x.transform(d.create_df_11())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="TwoColumnMethod transformer does not produce the expected output",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "TwoColumnOperatorTransformer"
