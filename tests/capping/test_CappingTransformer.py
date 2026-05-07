import narwhals as nw
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    EmptyCappingsFitTransformPassTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    OtherBaseBehaviourTestsNumeric,
)
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.capping import CappingTransformer


class TestInit(GenericCappingInitTests):
    """Tests for CappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestFit(GenericCappingFitTests):
    """Tests for CappingTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestTransform(GenericCappingTransformTests, GenericTransformTests):
    """Tests for CappingTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_expected_output_min_and_max_combinations(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
        from_json,
        lazy,
    ):
        """Test that capping is applied correctly in transform."""

        df = d.create_df_3(library=library)
        expected = self.expected_df_1(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [2, 5], "b": [None, 7], "c": [0, None]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(_convert_to_lazy(df, lazy))
        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy))

        print(df_transformed)
        print(expected)
        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy), expected)

        # Check outcomes for single rows
        df = nw.from_native(df)
        expected = nw.from_native(expected)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row,
            )


class TestLazyYSupport:
    """Tests for lazy y support in CappingTransformer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that CappingTransformer accepts LazyFrame for y parameter."""
        df_dict = {"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
        df = dataframe_init_dispatch(df_dict, library)

        y_lazy = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

        transformer = CappingTransformer(quantiles={"b": [0.1, 0.9]})

        # Fit should accept lazy y and not raise an error
        transformer.fit(df, y_lazy)

        # Transform should apply caps correctly based on learned quantiles
        expected = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.0, 2.0, 3.0, 4.0, 4.5],
            }
        )

        transformed = transformer.transform(df)

        assert_frame_equal_dispatch(transformed, expected)


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyCappingsFitTransformPassTests,
    OtherBaseBehaviourTestsNumeric,
):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    def test_get_params_call_with_capping_values_none(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test get_params method when capping_values is None."""
        args = minimal_attribute_dict[self.transformer_name]
        args["capping_values"] = None
        args["quantiles"] = {"a": [0.1, 0.9]}
        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Ensure no AttributeError is raised when calling get_params method
        try:
            transformer.get_params()
        except AttributeError as e:
            pytest.fail(f"AttributeError was raised: {e}")
