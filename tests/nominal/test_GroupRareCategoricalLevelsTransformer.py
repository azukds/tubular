import re

import narwhals as nw
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DummyWeightColumnMixinTests,
    EmptyColumnsFitTransformPassTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.nominal import GroupRareCategoricalLevelsTransformer


def create_test_df_with_invalid_weights(library="pandas"):
    """test df to use for weighted fit tests"""

    df_dict = {
        # the two rows on the end are invalid weights, so these
        # rows will be ignored
        "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3, -1, -20],
        "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", None, "i", "j"],
    }

    df = dataframe_init_dispatch(df_dict, library)

    return nw.from_native(df).with_columns(nw.col("c").cast(nw.Categorical)).to_native()


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for GroupRareCategoricalLevelsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareCategoricalLevelsTransformer"

    @pytest.mark.parametrize("cut_off_percent", ["a", -1.0, 2.0])
    def test_cut_off_percent_errors(self, cut_off_percent):
        """Test that an exception is raised if cut_off_percent is not a float between 0 and 1."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareCategoricalLevelsTransformer(
                columns="a", cut_off_percent=cut_off_percent
            )

    def test_record_rare_levels_not_bool_error(self):
        """Test that an exception is raised if record_rare_levels is not a bool."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareCategoricalLevelsTransformer(columns="a", record_rare_levels=2)

    def test_unseen_levels_to_rare_not_bool_error(self):
        """Test that an exception is raised if unseen_levels_to_rare is not a bool."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareCategoricalLevelsTransformer(columns="a", unseen_levels_to_rare=2)

    # overload this one until weight mixin is converted to beartype
    @pytest.mark.parametrize("weights_column", [0, ["a"], {"a": 10}])
    def test_weight_arg_errors(self, weights_column):
        """Test that appropriate errors are throw for bad weight arg."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareCategoricalLevelsTransformer(
                columns="a", weights_column=weights_column
            )


class TestFit(GenericFitTests, WeightColumnFitMixinTests, DummyWeightColumnMixinTests):
    """Tests for GroupRareCategoricalLevelsTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareCategoricalLevelsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_no_weight(self, library, lazy):
        """Test that the non rare levels learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("c").fill_null("c")).to_native()

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert transformer.non_rare_levels == {"c": ["a", "c", "e"]}
        assert transformer.rare_levels_record == {"c": ["f", "g", "h"]}

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight(self, library, lazy):
        """Test that the non rare levels learnt during fit, using a weight, are expected."""
        df = create_test_df_with_invalid_weights(library=library)

        df = nw.from_native(df).with_columns(nw.col("c").fill_null("f")).to_native()

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"],
            cut_off_percent=0.2,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert transformer.non_rare_levels == {"c": ["f", "g"]}

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("col", ["a", "b"])
    def test_column_type_error(self, col, library, lazy):
        """Test that error is raised if fit is run on non-categorical columns."""
        # b is string, a is numeric
        df = d.create_df_5(library=library)

        transformer = GroupRareCategoricalLevelsTransformer(columns=[col])

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        msg = "GroupRareCategoricalLevelsTransformer: transformer can only handle Categorical/Enum type columns"
        with pytest.raises(TypeError, match=msg):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_training_data_levels_stored(self, library, lazy):
        """Test that the levels present in the training data are stored if unseen_levels_to_rare is false"""
        df = d.create_df_8(library=library)

        expected_training_data_levels = {"c": sorted(set(df["c"]))}

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"], unseen_levels_to_rare=False
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert expected_training_data_levels == transformer.training_data_levels

    # NOTE - have decided not to include the test for failed fit here,
    # as in the case of fitting on an empty df finding no rare/non-rare levels
    # is not actually an invalid result


class TestTransform(GenericTransformTests):
    """Tests for GroupRareCategoricalLevelsTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareCategoricalLevelsTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_not_fitted_error_raised(self, library):
        """Test that transform raises NotFittedError if fit has not been run."""
        df = d.create_categorical_df_no_nulls(library=library)

        transformer = GroupRareCategoricalLevelsTransformer(columns=["b"])

        if _check_if_skip_test(transformer, df, lazy=False, from_json=False):
            return

        with pytest.raises(NotFittedError):
            transformer.transform(df)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_updated(self, library, from_json):
        """Test that the original dataframe is not modified when transform is used."""
        df = d.create_categorical_df_no_nulls(library=library)

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["b"], cut_off_percent=0.5
        )

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, d.create_categorical_df_no_nulls(library))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified(self, library, from_json, lazy):
        """Test that the non_rare_levels from fit are not changed in transform."""
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("c").fill_null("c")).to_native()

        transformer = GroupRareCategoricalLevelsTransformer(columns=["c"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        transformer2 = GroupRareCategoricalLevelsTransformer(columns=["c"])
        transformer2.fit(_convert_to_lazy(df, lazy=lazy))
        transformer2 = _handle_from_json(transformer2, from_json)
        transformer2.transform(_convert_to_lazy(df, lazy=lazy))

        assert transformer2.non_rare_levels == transformer.non_rare_levels

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_no_weight(self, library, from_json, lazy):
        """Test that the output is expected from transform."""
        df = d.create_df_5(library=library)

        df = (
            nw.from_native(df)
            .select("a", "c")
            .with_columns(nw.col("c").fill_null("e"))
            .to_native()
        )

        expected = dataframe_init_dispatch(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
                "c": ["a", "a", "c", "c", "e", "e", "rare", "rare", "rare", "e"],
            },
            library,
        )
        # output is an enum of the non rare levels, plus the rare level
        expected = (
            nw.from_native(expected)
            .with_columns(nw.col("c").cast(nw.Enum(["e", "c", "a", "rare"])))
            .to_native()
        )

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set learnt attrs directly rather than fitting, so test works with decorators
        transformer.non_rare_levels = {"c": ["e", "c", "a"]}
        transformer.rare_levels_record = {}
        transformer.training_data_levels = {}
        transformer = _handle_from_json(transformer, from_json)
        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(
            _collect_frame(df_transformed, lazy=lazy),
            expected,
            check_categorical=False,
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_output_is_enum(self, library, from_json, lazy):
        """Test that output columns are enum, with categories of non rare levels plus rare level."""
        df = d.create_df_8(library=library)

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"], cut_off_percent=0.25, rare_level_name="rare_level"
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))
        transformer = _handle_from_json(transformer, from_json)

        output_df = transformer.transform(_convert_to_lazy(df, lazy=lazy))
        output_df = nw.from_native(_collect_frame(output_df, lazy=lazy))

        assert isinstance(output_df.schema["c"], nw.Enum)
        assert output_df["c"].to_list() == ["a", "a", "rare_level", "rare_level", "a"]
        assert output_df["c"].cat.get_categories().to_list() == ["a", "rare_level"]

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("bad_col", ["a", "b"])
    def test_column_type_error(self, library, from_json, lazy, bad_col):
        """Test that error is raised if transform is run on non-categorical columns."""
        # b is string, a is numeric
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("c").fill_null("c")).to_native()

        transformer = GroupRareCategoricalLevelsTransformer(columns=["c"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        # overwrite columns to wrong type before transform, to trigger error
        transformer.columns = [bad_col]
        transformer = _handle_from_json(transformer, from_json)

        msg = re.escape(
            "GroupRareCategoricalLevelsTransformer: transformer can only handle Categorical/Enum type columns",
        )
        with pytest.raises(TypeError, match=msg):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_rare_categories_forgotten(self, library, from_json, lazy):
        "test that for category dtype, categories encoded as rare are forgotten by series"
        df = d.create_df_8(library=library)

        column = "c"

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=column,
            cut_off_percent=0.25,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))
        transformer = _handle_from_json(transformer, from_json)

        output_df = transformer.transform(_convert_to_lazy(df, lazy=lazy))
        output_df = _collect_frame(output_df, lazy=lazy)

        output_categories = (
            nw.from_native(output_df)[column].cat.get_categories().to_list()
        )

        for cat in ["c", "b"]:
            assert cat not in output_categories, (
                f"expected {cat} to be forgotten from column {column}"
            )


class TestLazyYSupport:
    """Tests for lazy y support in GroupRareCategoricalLevelsTransformer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that GroupRareCategoricalLevelsTransformer accepts LazyFrame for y parameter."""
        df_dict = {
            "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
            "c": ["a", "a", "a", "d", "e", "f", "g", "h", "i", "j"],
        }
        df = dataframe_init_dispatch(df_dict, library)
        df = nw.from_native(df).with_columns(nw.col("c").cast(nw.Categorical))
        df = df.to_native()

        y_lazy = pl.LazyFrame({"a": list(range(10))})

        transformer = GroupRareCategoricalLevelsTransformer(
            columns=["c"], cut_off_percent=0.2
        )

        transformer.fit(df, y_lazy)

        actual = transformer.transform(df)

        assert actual["c"].to_list() == ["a", "a", "a"] + ["rare"] * 7


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyColumnsFitTransformPassTests,
):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareCategoricalLevelsTransformer"
