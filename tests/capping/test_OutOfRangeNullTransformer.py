import narwhals as nw
import pytest

import tests.test_data as d
from tests.base_tests import OtherBaseBehaviourTests
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.capping import OutOfRangeNullTransformer


class TestInit(GenericCappingInitTests):
    """Tests for OutOfRangeNullTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OutOfRangeNullTransformer"


class TestFit(GenericCappingFitTests):
    """Tests for OutOfRangeNullTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OutOfRangeNullTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("values", "sample_weight", "quantiles"),
        [
            ([1, 2, 3], [1, 2, 1], [0.1, 0.5]),
            ([1, 2, 3], None, [0.1, 0.5]),
            ([2, 4, 6, 8], [3, 2, 1, 1], [None, 0.5]),
            ([2, 4, 6, 8], None, [None, 0.5]),
            ([-1, -5, -10, 20, 30], [1, 2, 2, 2, 2], [0.1, None]),
            ([-1, -5, -10, 20, 40], None, [0.1, None]),
        ],
    )
    def test_replacement_values_updated(
        self,
        values,
        sample_weight,
        quantiles,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that weighted_quantile gives the expected outputs."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": quantiles}
        args["capping_values"] = None
        args["weights_column"] = "w"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if not sample_weight:
            sample_weight = [1] * len(values)

        df_dict = {
            "a": values,
            "w": sample_weight,
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        transformer.fit(df)

        lower_replacement = None if quantiles[0] else False
        upper_replacement = None if quantiles[1] else False
        expected = [lower_replacement, upper_replacement]
        assert transformer._replacement_values["a"] == expected, (
            f"unexpected value for replacement_values attribute, expected {expected} but got {transformer._replacement_values}"
        )


class TestTransform(GenericCappingTransformTests):
    """Tests for OutOfRangeNullTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OutOfRangeNullTransformer"

    def expected_df_1(self, library="pandas"):
        """Expected output from test_expected_output_min_and_max."""

        df_dict = {
            "a": [None, 2, 3, 4, 5, None, None],
            "b": [1, 2, 3, None, 7, None, None],
            "c": [None, 1, 2, 3, None, None, None],
        }

        return dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_min_and_max_combinations(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that capping is applied correctly in transform."""

        df = d.create_df_3(library=library)
        expected = self.expected_df_1(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [2, 5], "b": [None, 7], "c": [0, None]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = transformer.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

        # Check outcomes for single rows
        df = nw.from_native(df)
        expected = nw.from_native(expected)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OutOfRangeNullTransformer"


class TestSetReplacementValues:
    """Test for the OutOfRangeNullTransformer.set_replacement_values() method."""

    @pytest.mark.parametrize(
        ("capping_values", "expected_replacement_values"),
        [
            (
                {"a": [0, 1], "b": [None, 1], "c": [3, None]},
                {"a": [None, None], "b": [False, None], "c": [None, False]},
            ),
            ({}, {}),
            ({"a": [None, 0.1]}, {"a": [False, None]}),
        ],
    )
    def test_expected_replacement_values_output(
        self,
        capping_values,
        expected_replacement_values,
    ):
        """Test the _replacement_values attribute is modified as expected given the prior values of the attribute."""

        replacement_values = OutOfRangeNullTransformer.set_replacement_values(
            capping_values,
        )

        assert replacement_values == expected_replacement_values, (
            f"set_replacement_values output not as expected, expected {expected_replacement_values} but got {replacement_values}"
        )
