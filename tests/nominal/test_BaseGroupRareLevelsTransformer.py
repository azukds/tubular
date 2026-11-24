import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from test_BaseNominalTransformer import (
    GenericNominalTransformTests,
)

from tests.base_tests import (
    ColumnStrListInitTests,
    DummyWeightColumnMixinTests,
    GenericFitTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.utils import (
    _check_if_skip_test,
    _convert_to_lazy,
    _handle_from_json,
    dataframe_init_dispatch,
)


def create_group_rare_levels_test_df_with_invalid_weights(library="pandas"):
    """test df to use for fit tests of GroupRareLevelsTransformer"""

    df_dict = {
        # the two rows on the end are invalid weights, so these
        # rows will be ignored
        "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3, -1, -20],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None, "z", "x"],
        "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", None, "i", "j"],
    }

    df = dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


def create_group_rare_levels_test_df(library="pandas"):
    """test df to use for fit tests of GroupRareLevelsTransformer"""

    df_dict = {
        "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None],
        "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", None],
    }

    df = dataframe_init_dispatch(df_dict, library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return df.to_native()


class BaseGroupRareLevelsTransformerInitTests(
    ColumnStrListInitTests, WeightColumnInitMixinTests
):
    """Tests for GroupRareLevelsTransformer.init()."""

    def test_cut_off_percent_not_float_error(self, uninitialized_transformers):
        """Test that an exception is raised if cut_off_percent is not an float."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", cut_off_percent="a"
            )

    def test_cut_off_percent_negative_error(self, uninitialized_transformers):
        """Test that an exception is raised if cut_off_percent is negative."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", cut_off_percent=-1.0
            )

    def test_cut_off_percent_gt_one_error(self, uninitialized_transformers):
        """Test that an exception is raised if cut_off_percent is greater than 1."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", cut_off_percent=2.0
            )

    def test_record_rare_levels_not_bool_error(self, uninitialized_transformers):
        """Test that an exception is raised if record_rare_levels is not a bool."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", record_rare_levels=2
            )

    def test_unseen_levels_to_rare_not_bool_error(self, uninitialized_transformers):
        """Test that an exception is raised if unseen_levels_to_rare is not a bool."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", unseen_levels_to_rare=2
            )

    # overload this one until weight mixin is converted to beartype
    @pytest.mark.parametrize("weights_column", [0, ["a"], {"a": 10}])
    def test_weight_arg_errors(self, weights_column, uninitialized_transformers):
        """Test that appropriate errors are throw for bad weight arg."""

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](
                columns="a", weights_column=weights_column
            )


class BaseGroupRareLevelsTransformerFitTests(
    GenericFitTests, WeightColumnFitMixinTests, DummyWeightColumnMixinTests
):
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_training_data_levels_stored(
        self, lazy, uninitialized_transformers, minimal_dataframe_lookup
    ):
        """Test that the levels present in the training data are stored if unseen_levels_to_rare is false"""
        df = minimal_dataframe_lookup[self.transformer_name]

        expected_training_data_levels = {
            "b": sorted(set(df["b"])),
        }

        transformer = uninitialized_transformers[self.transformer_name](
            columns=["b"], unseen_levels_to_rare=False
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert expected_training_data_levels == transformer.training_data_levels, (
            "Training data values not correctly stored when unseen_levels_to_rare is false"
        )

    # NOTE - have decided not to include the test for failed fit here,
    # as in the case of fitting on an empty df finding no rare/non-rare levels
    # is not actually an invalid result
    # (compare this to an imputer, were finding None as the imputation value
    # is nonsensical)


class BaseGroupRareLevelsTransformerTransformTests(GenericNominalTransformTests):
    def test_non_mappable_rows_exception_raised(self):
        """override test in GenericNominalTransformTests as not relevant to this transformer."""

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_learnt_values_not_modified(
        self, from_json, lazy, uninitialized_transformers, minimal_dataframe_lookup
    ):
        """Test that the non_rare_levels from fit are not changed in transform."""
        df = minimal_dataframe_lookup[self.transformer_name]

        transformer = uninitialized_transformers[self.transformer_name](columns=["b"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        transformer2 = uninitialized_transformers[self.transformer_name](columns=["b"])

        transformer2.fit(_convert_to_lazy(df, lazy=lazy))
        transformer2 = _handle_from_json(transformer2, from_json)
        transformer2.transform(_convert_to_lazy(df, lazy=lazy))

        actual = transformer2.non_rare_levels
        expected = transformer.non_rare_levels

        assert actual == expected, (
            f"non_rare_levels attr modified in transform, expected {expected} but got {actual}"
        )
