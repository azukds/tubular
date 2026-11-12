import re

import narwhals as nw
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from test_BaseNominalTransformer import (
    GenericNominalTransformTests,
)

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DummyWeightColumnMixinTests,
    EmptyColumnsFitTransformPassTests,
    GenericFitTests,
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
from tubular.nominal import GroupRareStringLevelsTransformer


def create_test_df_with_invalid_weights(library="pandas"):
    """test df to use for weighted fit tests"""

    df_dict = {
        # the two rows on the end are invalid weights, so these
        # rows will be ignored
        "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3, -1, -20],
        "b": ["a", "a", "a", "d", "e", "f", "g", None, None, None, "z", "x"],
    }

    return dataframe_init_dispatch(df_dict, library)


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for GroupRareStringLevelsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"

    @pytest.mark.parametrize("cut_off_percent", ["a", -1.0, 2.0])
    def test_cut_off_percent_errors(self, cut_off_percent):
        """Test that an exception is raised if cut_off_percent is not a float between 0 and 1."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareStringLevelsTransformer(
                columns="a", cut_off_percent=cut_off_percent
            )

    def test_record_rare_levels_not_bool_error(self):
        """Test that an exception is raised if record_rare_levels is not a bool."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareStringLevelsTransformer(columns="a", record_rare_levels=2)

    def test_unseen_levels_to_rare_not_bool_error(self):
        """Test that an exception is raised if unseen_levels_to_rare is not a bool."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareStringLevelsTransformer(columns="a", unseen_levels_to_rare=2)

    # overload this one until weight mixin is converted to beartype
    @pytest.mark.parametrize("weights_column", [0, ["a"], {"a": 10}])
    def test_weight_arg_errors(self, weights_column):
        """Test that appropriate errors are throw for bad weight arg."""
        with pytest.raises(BeartypeCallHintParamViolation):
            GroupRareStringLevelsTransformer(columns="a", weights_column=weights_column)


class TestFit(GenericFitTests, WeightColumnFitMixinTests, DummyWeightColumnMixinTests):
    """Tests for GroupRareStringLevelsTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_no_weight(self, library, lazy):
        """Test that the non rare levels learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert transformer.non_rare_levels == {"b": ["a"]}
        assert transformer.rare_levels_record == {"b": ["d", "e", "f", "g"]}

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight(self, library, lazy):
        """Test that the non rare levels learnt during fit, using a weight, are expected."""
        df = create_test_df_with_invalid_weights(library=library)

        df = nw.from_native(df).with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert transformer.non_rare_levels == {"b": ["a"]}

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("col", ["a", "c"])
    def test_column_type_error(self, col, library, lazy):
        """Test that error is raised if fit is run on non-string columns."""
        # c is categorical, a is numeric
        df = d.create_df_5(library=library)

        transformer = GroupRareStringLevelsTransformer(columns=[col])

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        msg = "GroupRareStringLevelsTransformer: transformer can only handle String type columns"
        with pytest.raises(TypeError, match=msg):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_training_data_levels_stored(self, library, lazy):
        """Test that the levels present in the training data are stored if unseen_levels_to_rare is false"""
        df = d.create_df_8(library=library)

        expected_training_data_levels = {"b": sorted(set(df["b"]))}

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], unseen_levels_to_rare=False
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        assert expected_training_data_levels == transformer.training_data_levels

    # NOTE - have decided not to include the test for failed fit here,
    # as in the case of fitting on an empty df finding no rare/non-rare levels
    # is not actually an invalid result


class TestTransform(GenericNominalTransformTests):
    """Tests for GroupRareStringLevelsTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"

    def test_non_mappable_rows_exception_raised(self):
        """override test in GenericNominalTransformTests as not relevant to this transformer."""

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified(self, library, from_json, lazy):
        """Test that the non_rare_levels from fit are not changed in transform."""
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareStringLevelsTransformer(columns=["b"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        transformer2 = GroupRareStringLevelsTransformer(columns=["b"])
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
            .select("a", "b")
            .with_columns(nw.col("b").fill_null("a"))
            .to_native()
        )

        expected = dataframe_init_dispatch(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
                "b": ["a", "a", "a", "rare", "rare", "rare", "rare", "a", "a", "a"],
            },
            library,
        )

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set learnt attrs directly rather than fitting, so test works with decorators
        transformer.non_rare_levels = {"b": ["a"]}
        transformer.rare_levels_record = {}
        transformer.training_data_levels = {}
        transformer = _handle_from_json(transformer, from_json)
        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_weight(self, library, from_json, lazy):
        """Test that the output is expected from transform, when weights are used."""
        df = create_test_df_with_invalid_weights(library=library).head(10)

        df = nw.from_native(df).with_columns(nw.col("b").fill_null("a")).to_native()

        expected = dataframe_init_dispatch(
            {
                "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
                "b": ["a", "a", "a", "rare", "rare", "rare", "rare", "a", "a", "a"],
            },
            library,
        )

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.non_rare_levels = {"b": ["a"]}
        transformer.rare_levels_record = {}
        transformer.training_data_levels = {}
        transformer = _handle_from_json(transformer, from_json)
        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("bad_col", ["a", "c"])
    def test_column_type_error(self, library, from_json, lazy, bad_col):
        """Test that error is raised if transform is run on non-string columns."""
        # c is categorical, a is numeric
        df = d.create_df_5(library=library)

        df = nw.from_native(df).with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareStringLevelsTransformer(columns=["b"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        # overwrite columns to wrong type before transform, to trigger error
        transformer.columns = [bad_col]
        transformer = _handle_from_json(transformer, from_json)

        msg = re.escape(
            "GroupRareStringLevelsTransformer: transformer can only handle String type columns",
        )
        with pytest.raises(TypeError, match=msg):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_unseen_levels_not_encoded(self, library, from_json, lazy):
        """Test that unseen levels are not encoded when unseen_levels_to_rare is false"""
        df = d.create_df_8(library=library)

        expected = ["w", "w", "rare", "rare", "unseen_level"]

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            unseen_levels_to_rare=False,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        df = nw.from_native(_convert_to_lazy(df, lazy=lazy))
        native_backend = nw.get_native_namespace(df)

        df = df.with_columns(
            nw.new_series(
                name="b",
                values=["w", "w", "z", "y", "unseen_level"],
                backend=native_backend,
            ),
        ).to_native()

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        actual = list(_collect_frame(df_transformed, lazy=lazy)["b"])

        assert actual == expected

    @pytest.mark.parametrize("from_json", [True, False])
    def test_unseen_levels_to_rare(self, from_json):
        """Test that unseen levels are encoded as rare when unseen_levels_to_rare is true"""
        df = d.create_df_8(library="polars")

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], cut_off_percent=0.3
        )
        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        new_df = pl.DataFrame({"b": ["w", "unseen_level"]})

        actual = transformer.transform(new_df)["b"].to_list()

        assert actual == ["w", "rare"]


class TestLazyYSupport:
    """Tests for lazy y support in GroupRareStringLevelsTransformer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that GroupRareStringLevelsTransformer accepts LazyFrame for y parameter."""
        df_dict = {
            "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
            "b": ["a", "a", "a", "d", "e", "f", "g", "h", "i", "j"],
        }
        df = dataframe_init_dispatch(df_dict, library)

        y_lazy = pl.LazyFrame({"a": list(range(10))})

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], cut_off_percent=0.2
        )

        transformer.fit(df, y_lazy)

        expected = dataframe_init_dispatch(
            {
                "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
                "b": ["a", "a", "a"] + ["rare"] * 7,
            },
            library,
        )

        assert_frame_equal_dispatch(transformer.transform(df), expected)


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
        cls.transformer_name = "GroupRareStringLevelsTransformer"
