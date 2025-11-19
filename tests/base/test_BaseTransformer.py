import narwhals as nw
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestTransform(GenericTransformTests, ReturnNativeTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize(
        "return_native",
        [True, False],
    )
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_X_returned(
        self,
        minimal_dataframe_lookup,
        uninitialized_transformers,
        minimal_attribute_dict,
        return_native,
        lazy,
        from_json,
    ):
        """Test that X is returned from transform."""
        df = minimal_dataframe_lookup[self.transformer_name]
        args = minimal_attribute_dict[self.transformer_name]
        args["return_native"] = return_native
        x = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(x, df, lazy=lazy, from_json=from_json):
            return

        df = nw.from_native(df)
        expected = df.clone()

        df = nw.to_native(df)
        expected = nw.to_native(expected)

        x = _handle_from_json(x, from_json)

        df_transformed = x.transform(X=_convert_to_lazy(df, lazy))

        if not x.return_native:
            df_transformed = nw.to_native(df_transformed)

        assert_frame_equal_dispatch(
            expected,
            _collect_frame(df_transformed, lazy),
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"
