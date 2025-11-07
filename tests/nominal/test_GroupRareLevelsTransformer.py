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
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.nominal import GroupRareLevelsTransformer


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
    @pytest.mark.parametrize("weights_column", (0, ["a"], {"a": 10}))
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

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_no_weight(self, library):
        """Test that the impute values learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("c"),
        ).to_native()

        x = GroupRareLevelsTransformer(columns=["b", "c"], cut_off_percent=0.2)

        x.fit(df)

        expected = {"b": ["a"], "c": ["a", "c", "e"]}
        actual = x.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight(self, library):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = d.create_df_6(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        x = GroupRareLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        x.fit(df)

        expected = {"b": ["a"]}
        actual = x.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight_2(self, library):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = d.create_df_6(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").fill_null("f")).to_native()

        x = GroupRareLevelsTransformer(
            columns=["c"],
            cut_off_percent=0.2,
            weights_column="a",
        )

        x.fit(df)

        expected = {"c": ["f", "g"]}
        actual = x.non_rare_levels
        assert actual == expected, (
            f"non_rare_levels attribute not fit as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("col", ["a", "c"])
    def test_column_strlike_error(self, col, library):
        """Test that checks error is raised if transform is run on non-strlike columns."""
        df = d.create_df_10(library=library)

        x = GroupRareLevelsTransformer(columns=[col], rare_level_name="bla")

        msg = "GroupRareLevelsTransformer: transformer must run on str-like columns"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            x.fit(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_training_data_levels_stored(self, library):
        """Test that the levels present in the training data are stored if unseen_levels_to_rare is false"""
        df = d.create_df_8(library=library)

        expected_training_data_levels = {
            "b": sorted(set(df["b"])),
            "c": sorted(set(df["c"])),
        }

        x = GroupRareLevelsTransformer(columns=["b", "c"], unseen_levels_to_rare=False)
        x.fit(df)

        assert expected_training_data_levels == x.training_data_levels, (
            "Training data values not correctly stored when unseen_levels_to_rare is false"
        )


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

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified(self, library, from_json):
        """Test that the non_rare_levels from fit are not changed in transform."""
        df = d.create_df_5(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("c"),
        ).to_native()

        x = GroupRareLevelsTransformer(columns=["b", "c"])

        x.fit(df)
        x = _handle_from_json(x, from_json)

        x2 = GroupRareLevelsTransformer(columns=["b", "c"])

        x2.fit(df)
        x2 = _handle_from_json(x2, from_json)
        x2.transform(df)

        actual = x2.non_rare_levels
        expected = x.non_rare_levels

        assert actual == expected, (
            f"non_rare_levels attr modified in transform, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_no_weight(self, library, from_json):
        """Test that the output is expected from transform."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("e"),
        ).to_native()

        expected = self.expected_df_1(library=library)

        x = GroupRareLevelsTransformer(columns=["b", "c"], cut_off_percent=0.2)

        # set the mappging dict directly rather than fitting x on df so test works with decorators
        x.non_rare_levels = {"b": ["a"], "c": ["e", "c", "a"]}
        x = _handle_from_json(x, from_json)
        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected, check_categorical=False)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_weight(self, library, from_json):
        """Test that the output is expected from transform, when weights are used."""

        df = d.create_df_6(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        expected = self.expected_df_2(library=library)

        x = GroupRareLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        # set the mapping dict directly rather than fitting x on df so test works with decorators
        x.non_rare_levels = {"b": ["a"]}
        x = _handle_from_json(x, from_json)
        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_column_strlike_error(self, library, from_json):
        """Test that checks error is raised if transform is run on non-strlike columns."""
        df = d.create_df_10(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        x = GroupRareLevelsTransformer(columns=["b"], rare_level_name="bla")

        x.fit(df)
        # overwrite columns to non str-like before transform, to trigger error
        x.columns = ["a"]
        x = _handle_from_json(x, from_json)

        msg = re.escape(
            "GroupRareLevelsTransformer: transformer must run on str-like columns, but got non str-like {'a'}",
        )
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            x.transform(df)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_unseen_levels_not_encoded(self, library, from_json):
        """Test that unseen levels are not encoded when unseen_levels_to_rare is false"""

        df = d.create_df_8(library=library)

        expected = ["w", "w", "rare", "rare", "unseen_level"]

        x = GroupRareLevelsTransformer(
            columns=["b", "c"],
            cut_off_percent=0.3,
            unseen_levels_to_rare=False,
        )
        x.fit(df)
        x = _handle_from_json(x, from_json)

        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)

        df = df.with_columns(
            nw.new_series(
                name="b",
                values=["w", "w", "z", "y", "unseen_level"],
                backend=native_backend,
            ),
        ).to_native()

        df_transformed = x.transform(df)

        actual = list(df_transformed["b"])

        assert actual == expected, (
            f"unseen level handling not working as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize("from_json", ["True", "False"])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_rare_categories_forgotten(self, library, from_json):
        "test that for category dtype, categories encoded as rare are forgotten by series"

        df = d.create_df_8(library=library)

        column = "c"

        x = GroupRareLevelsTransformer(
            columns=column,
            cut_off_percent=0.25,
        )

        expected_removed_cats = ["c", "b"]

        x.fit(df)
        x = _handle_from_json(x, from_json)
        output_df = x.transform(df)

        output_categories = (
            nw.from_native(output_df)[column].cat.get_categories().to_list()
        )

        for cat in expected_removed_cats:
            assert cat not in output_categories, (
                f"{x.classname} output columns should forget rare encoded categories, expected {cat} to be forgotten from column {column}"
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"
