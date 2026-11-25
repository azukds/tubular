import re

import narwhals as nw
import pandas as pd
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
    create_group_rare_levels_test_df_with_invalid_weights,
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


class TestInit(BaseGroupRareLevelsTransformerInitTests):
    """Tests for GroupRareLevelsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"


class TestFit(BaseGroupRareLevelsTransformerFitTests):
    """Tests for GroupRareLevelsTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "GroupRareLevelsTransformer"

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

        msg = f"{self.transformer_name}: transformer can only handle String/Categorical/Enum type columns"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_weight_2(self, library, lazy, uninitialized_transformers):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = create_group_rare_levels_test_df_with_invalid_weights(library=library)

        # handle nulls
        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").fill_null("f")).to_native()

        transformer = uninitialized_transformers[self.transformer_name](
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
    def test_learnt_values_no_weight(self, library, lazy, uninitialized_transformers):
        """Test that the impute values learnt during fit, without using a weight, are expected."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("c"),
        ).to_native()

        transformer = uninitialized_transformers[self.transformer_name](
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
    def test_learnt_values_weight(self, library, lazy, uninitialized_transformers):
        """Test that the impute values learnt during fit, using a weight, are expected."""
        df = create_group_rare_levels_test_df_with_invalid_weights(library=library)

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

        return (
            nw.from_native(df)
            .with_columns(
                nw.col("c").cast(nw.Categorical),
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

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("cat_type", ["Categorical", "Enum"])
    def test_expected_output_no_weight(self, library, from_json, lazy, cat_type):
        """Test that the output is expected from transform."""
        df = d.create_df_5(library=library)

        # first handle nulls
        df = nw.from_native(df)
        df = df.with_columns(
            nw.col("b").fill_null("a"),
            nw.col("c").fill_null("e"),
        )

        if cat_type == "Enum":
            categories = df["c"].unique()
            df = df.with_columns(nw.col("c").cast(nw.Enum(categories=categories)))
        df = df.to_native()

        expected = self.expected_df_1(library=library)

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"], cut_off_percent=0.2
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        # set the mappging dict directly rather than fitting x on df so test works with decorators
        transformer.non_rare_levels = {"b": ["a"], "c": ["e", "c", "a"]}
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
        df = df.with_columns(nw.col("b").fill_null("a"))
        df = df.to_native()

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

        transformer = GroupRareLevelsTransformer(columns=["b"], rare_level_name="bla")

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)
        # overwrite columns to non str-like before transform, to trigger error
        transformer.columns = ["a"]
        transformer = _handle_from_json(transformer, from_json)

        msg = re.escape(
            "GroupRareLevelsTransformer: transformer can only handle String/Categorical/Enum type columns",
        )
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("cat_type", ["Categorical", "Enum"])
    def test_expected_output_unseen_levels_not_encoded(
        self, library, from_json, lazy, cat_type
    ):
        """Test that unseen levels are not encoded when unseen_levels_to_rare is false"""

        df = d.create_df_8(library=library)
        df = nw.from_native(df)
        if cat_type == "Enum":
            categories = df["c"].unique()
            df = df.with_columns(nw.col("c").cast(nw.Enum(categories=categories)))
        df = df.to_native()

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
    @pytest.mark.parametrize("cat_type", ["Categorical", "Enum"])
    def test_rare_categories_forgotten(self, library, from_json, lazy, cat_type):
        "test that for category dtype, categories encoded as rare are forgotten by series"

        df = d.create_df_8(library=library)
        column = "c"
        df = nw.from_native(df)
        if cat_type == "Enum":
            categories = df[column].unique()
            df = df.with_columns(nw.col(column).cast(nw.Enum(categories=categories)))
        df = df.to_native()

        transformer = GroupRareLevelsTransformer(
            columns=column,
            cut_off_percent=0.25,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        expected_removed_cats = ["c", "b"]

        transformer.fit(_convert_to_lazy(df, lazy=lazy))
        transformer = _handle_from_json(transformer, from_json)

        output_df = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        output_df = _collect_frame(output_df, lazy=lazy)

        output_categories = (
            nw.from_native(output_df)[column].cat.get_categories().to_list()
        )

        for cat in expected_removed_cats:
            assert cat not in output_categories, (
                f"{transformer.classname} output columns should forget rare encoded categories, expected {cat} to be forgotten from column {column}"
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_handling_all_str_like_types_pandas(
        self, lazy: bool, from_json: bool
    ) -> None:
        """Test that the transformer can handle any str-like type in pandas."""
        df = pd.DataFrame(
            {
                "str": ["a", "b", "a", "a"],
                "categorical": ["a", "b", "a", "a"],
            },
        )
        df["str"] = df["str"].astype("string")
        df["categorical"] = df["categorical"].astype("category")
        transformer = GroupRareLevelsTransformer(
            columns=list(df.columns),
            cut_off_percent=0.3,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = transformer.fit(df)
        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))
        expected = pd.DataFrame(
            {
                "str": ["a", "rare", "a", "a"],
                "categorical": ["a", "rare", "a", "a"],
            },
        )
        expected["str"] = expected["str"].astype("string")
        expected["categorical"] = expected["categorical"].astype("category")

        assert_frame_equal_dispatch(
            _collect_frame(df_transformed, lazy=False),
            expected,
            check_categorical=False,
        )


class TestLazyYSupport:
    """Tests for lazy y support in GroupRareLevelsTransformer."""

    @pytest.mark.parametrize("library", ["polars"])
    def test_lazy_y_accepted(self, library):
        """Test that GroupRareLevelsTransformer accepts LazyFrame for y parameter."""
        df_dict = {
            "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
            "b": ["a", "a", "a", "d", "e", "f", "g", "h", "i", "j"],
            "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", "k"],
        }
        df = dataframe_init_dispatch(df_dict, library)

        y_lazy = pl.LazyFrame({"a": list(range(10))})

        transformer = GroupRareLevelsTransformer(
            columns=["b", "c"], cut_off_percent=0.2
        )

        # Fit should accept lazy y and not raise an error
        transformer.fit(df, y_lazy)

        # Transform should group rare levels correctly
        expected = dataframe_init_dispatch(
            {
                "a": [2, 2, 2, 2, 0, 2, 2, 2, 3, 3],
                "b": [
                    "a",
                    "a",
                    "a",
                    "rare",
                    "rare",
                    "rare",
                    "rare",
                    "rare",
                    "rare",
                    "rare",
                ],
                "c": ["rare", "rare", "rare", "rare", "f", "f", "f", "g", "g", "rare"],
            },
            library,
        )

        transformed = transformer.transform(df)

        assert_frame_equal_dispatch(transformed, expected)


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
        cls.transformer_name = "GroupRareLevelsTransformer"
