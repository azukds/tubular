import re

import narwhals as nw
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    EmptyColumnsFitTransformPassTests,
    OtherBaseBehaviourTests,
)
from tests.nominal.test_BaseGroupRareLevelsTransformer import (
    BaseGroupRareLevelsTransformerFitTests,
    BaseGroupRareLevelsTransformerInitTests,
    BaseGroupRareLevelsTransformerTransformTests,
    create_group_rare_levels_test_df,
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


class TestInit(BaseGroupRareLevelsTransformerInitTests):
    """Tests for GroupRareStringLevelsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"


class TestFit(BaseGroupRareLevelsTransformerFitTests):
    """Tests for GroupRareStringLevelsTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("col", ["a", "c"])
    def test_column_strlike_error(self, col, library, lazy, uninitialized_transformers):
        """Test that checks error is raised if transform is run on non-strlike columns."""
        df = d.create_df_10(library=library)

        transformer = uninitialized_transformers[self.transformer_name](
            columns=[col], rare_level_name="bla"
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        msg = (
            f"{self.transformer_name}: transformer can only handle String type columns"
        )
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_no_weight(self, library, lazy, uninitialized_transformers):
        """Test that the impute values learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
        ).to_native()

        transformer = uninitialized_transformers[self.transformer_name](
            columns=["b"], cut_off_percent=0.2
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
    def test_learnt_values_weight(self, library, lazy, uninitialized_transformers):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = create_test_df_with_invalid_weights(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").fill_null("a")).to_native()

        transformer = uninitialized_transformers[self.transformer_name](
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


class TestTransform(BaseGroupRareLevelsTransformerTransformTests):
    """Tests for GroupRareStringLevelsTransformer.transform()."""

    def expected_df_1(self, library="pandas"):
        """Expected output for test_expected_output_no_weight."""

        df = d.create_df_5(library=library)

        df = nw.from_native(df)
        return df.with_columns(
            nw.new_series(
                name="b",
                values=["a", "a", "a", "rare", "rare", "rare", "rare", "a", "a", "a"],
                backend=library,
            )
        ).to_native()

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

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareStringLevelsTransformer"

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
        ).to_native()

        expected = self.expected_df_1(library=library)

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set the mappging dict directly rather than fitting x on df so test works with decorators
        transformer.non_rare_levels = {"b": ["a"]}
        transformer.rare_levels_record = {}
        transformer.training_data_levels = {}
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

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"],
            cut_off_percent=0.3,
            weights_column="a",
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set the mapping dict directly rather than fitting x on df so test works with decorators
        transformer.non_rare_levels = {"b": ["a"]}
        transformer.rare_levels_record = {}
        transformer.training_data_levels = {}
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

        transformer = GroupRareStringLevelsTransformer(
            columns=["b"], rare_level_name="bla"
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        # overwrite columns to non str-like before transform, to trigger error
        transformer.columns = ["a"]
        transformer = _handle_from_json(transformer, from_json)

        msg = re.escape(
            "GroupRareStringLevelsTransformer: transformer can only handle String type columns but got columns with types {'a': Float64}",
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

        assert actual == expected, (
            f"unseen level handling not working as expected, expected {expected} but got {actual}"
        )


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
