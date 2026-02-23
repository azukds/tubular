import re

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from test_BaseNominalTransformer import GenericNominalTransformTests

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DummyWeightColumnMixinTests,
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
from tubular.nominal import GroupRareLevelsTransformer


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


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for GroupRareLevelsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"

    def test_cut_off_percent_not_float_error(self):
        """Test that an exception is raised if cut_off_percent is not an float."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", cut_off_percent="a")

    def test_cut_off_percent_negative_error(self):
        """Test that an exception is raised if cut_off_percent is negative."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", cut_off_percent=-1.0)

    def test_cut_off_percent_gt_one_error(self):
        """Test that an exception is raised if cut_off_percent is greater than 1."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", cut_off_percent=2.0)

    def test_record_rare_levels_not_bool_error(self):
        """Test that an exception is raised if record_rare_levels is not a bool."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", record_rare_levels=2)

    def test_unseen_levels_to_rare_not_bool_error(self):
        """Test that an exception is raised if unseen_levels_to_rare is not a bool."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", unseen_levels_to_rare=2)

    # overload this one until weight mixin is converted to beartype
    @pytest.mark.parametrize("weights_column", [0, ["a"], {"a": 10}])
    def test_weight_arg_errors(
        self,
        weights_column,
    ):
        """Test that appropriate errors are throw for bad weight arg."""

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            GroupRareLevelsTransformer(columns="a", weights_column=weights_column)


class TestFit(GenericFitTests, WeightColumnFitMixinTests, DummyWeightColumnMixinTests):
    """Tests for GroupRareLevelsTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_no_weight(self, library, lazy):
        """Test that the impute values learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("c"),
        ).to_native()

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        expected = {"b": ["a"], "c": ["a", "c", "e"]}
        actual = transformer.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight(self, library, lazy):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = create_group_rare_levels_test_df_with_invalid_weights(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        expected = {"b": ["a"]}
        actual = transformer.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight_2(self, library, lazy):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = create_group_rare_levels_test_df_with_invalid_weights(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").fill_null("f")).to_native()

        transformer = GroupRareLevelsTransformer(
            columns=["c"],
            cut_off_percent=0.2,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

        expected = {"c": ["f", "g"]}
        actual = transformer.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("col", ["a", "c"])
    def test_column_strlike_error(self, col, library, lazy):
        """Test that checks error is raised if transform is run on non-strlike columns."""
        df = d.create_df_10(library=library)

        transformer = GroupRareLevelsTransformer(columns=[col], rare_level_name="bla")

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        msg = "GroupRareLevelsTransformer: transformer must run on str-like columns"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_training_data_levels_stored(self, library, lazy):
        """Test that the levels present in the training data are stored if unseen_levels_to_rare is false"""
        df = d.create_df_8(library=library)

        expected_training_data_levels = {
            "b": sorted(set(df["b"])),
            "c": sorted(set(df["c"])),
        }

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"], unseen_levels_to_rare=False
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


class TestTransform(GenericNominalTransformTests):
    """Tests for GroupRareLevelsTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"

    def expected_df_1(self, library="pandas"):
        """Expected output for test_expected_output_no_weight."""

        df_dict = {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
            "b": ["a", "a", "a", "rare", "rare", "rare", "rare", "a", "a", "a"],
            "c": ["a", "a", "c", "c", "e", "e", "rare", "rare", "rare", "e"],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        # set categories for enum
        categories = ["e", "c", "a", "rare"]

        return (
            nw.from_native(df)
            .with_columns(
                nw.col("c").cast(nw.Enum(categories=categories)),
            )
            .to_native()
        )

    def expected_df_2(self, library="pandas"):
        """Expected output for test_expected_output_weight."""

        df_dict = {
            "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
            "b": ["a", "a", "a", "rare", "rare", "rare", "rare", "a", "a", "a"],
            "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", None],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)

        return df.with_columns(nw.col("c").cast(nw.Categorical)).to_native()

    def test_non_mappable_rows_exception_raised(self):
        """override test in GenericNominalTransformTests as not relevant to this transformer."""

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified(self, library, from_json, lazy):
        """Test that the non_rare_levels from fit are not changed in transform."""
        df = d.create_df_5(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("c"),
        ).to_native()

        transformer = GroupRareLevelsTransformer(columns=["b", "c"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        transformer2 = GroupRareLevelsTransformer(columns=["b", "c"])

        transformer2.fit(_convert_to_lazy(df, lazy=lazy))
        transformer2 = _handle_from_json(transformer2, from_json)
        transformer2.transform(_convert_to_lazy(df, lazy=lazy))

        actual = transformer2.non_rare_levels
        expected = transformer.non_rare_levels

        assert actual == expected, (
            f"non_rare_levels attr modified in transform, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_no_weight(self, library, from_json, lazy):
        """Test that the output is expected from transform."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("e"),
        ).to_native()

        expected = self.expected_df_1(library=library)

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set the mappging dict directly rather than fitting x on df so test works with decorators
        transformer.non_rare_levels = {"b": ["a"], "c": ["e", "c", "a"]}
        transformer.rare_levels_record_ = {}
        transformer = _handle_from_json(transformer, from_json)
        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(
            _collect_frame(df_transformed, lazy=lazy), expected, check_categorical=False
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_weight(self, library, from_json, lazy):
        """Test that the output is expected from transform, when weights are used."""

        df = create_group_rare_levels_test_df(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        expected = self.expected_df_2(library=library)

        transformer = GroupRareLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set the mapping dict directly rather than fitting x on df so test works with decorators
        transformer.non_rare_levels = {"b": ["a"]}
        transformer.rare_levels_record_ = {}
        transformer = _handle_from_json(transformer, from_json)
        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_column_strlike_error(self, library, from_json, lazy):
        """Test that checks error is raised if transform is run on non-strlike columns."""
        df = d.create_df_10(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = GroupRareLevelsTransformer(columns=["b"], rare_level_name="bla")

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        # overwrite columns to non str-like before transform, to trigger error
        transformer.columns = ["a"]
        transformer = _handle_from_json(transformer, from_json)

        msg = re.escape(
            "GroupRareLevelsTransformer: transformer must run on str-like columns, but got non str-like {'a'}",
        )
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_unseen_levels_not_encoded(self, library, from_json, lazy):
        """Test that unseen levels are not encoded when unseen_levels_to_rare is false"""

        df = d.create_df_8(library=library)

        expected = ["w", "w", "rare", "rare", "unseen_level"]

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"],
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

        assert actual == expected, (
            f"unseen level handling not working as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", ["True", "False"])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_rare_categories_forgotten(self, library, from_json, lazy):
        "test that for category dtype, categories encoded as rare are forgotten by series"

        df = d.create_df_8(library=library)

        column = "c"

        transformer = GroupRareLevelsTransformer(
            columns=column,
            cut_off_percent=0.25,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        expected_removed_cats = ["c", "b"]

        transformer.fit(_convert_to_lazy(df, lazy=lazy))
        transformer = _handle_from_json(transformer, from_json)
        output_df = transformer.transform(_collect_frame(df, lazy=lazy))

        output_df = _collect_frame(output_df, lazy=lazy)

        output_categories = (
            nw.from_native(output_df)[column].cat.get_categories().to_list()
        )

        for cat in expected_removed_cats:
            assert cat not in output_categories, (
                f"{transformer.classname} output columns should forget rare encoded categories, expected {cat} to be forgotten from column {column}"
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"
