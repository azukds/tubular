from itertools import product

import narwhals as nw
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    ColumnStrListInitTests,
    DummyWeightColumnMixinTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.mapping import BaseMappingTransformer
from tubular.nominal import MeanResponseTransformer


# Dataframe used exclusively in this testing script
def create_MeanResponseTransformer_test_df(library="pandas"):
    """Create DataFrame to use MeanResponseTransformer tests that correct values are.

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": ["a", "b", "c", "d", "e", "f"],
        "c": ["a", "b", "c", "d", "e", "f"],
        "d": [1, 2, 3, 4, 5, 6],
        "e": [1, 2, 3, 4, 5, 6],
        "f": [False, False, False, True, True, True],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


# Dataframe used exclusively in this testing script
def create_MeanResponseTransformer_test_df_unseen_levels(library="pandas"):
    """Create DataFrame to use in MeanResponseTransformer tests that check correct values are
    generated when using transform method on data with unseen levels.
    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1, 2, 3, 4, 5, 6, 7, 8],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [False, False, False, True, True, True, True, False],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


@pytest.fixture()
def learnt_mapping_dict():
    full_dict = {}

    b_dict = {
        "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
        "b_blue": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0},
        "b_yellow": {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0, "e": 0.0, "f": 0.0},
        "b_green": {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 1.0, "f": 1.0},
    }

    # c matches b, but is categorical (see create_MeanResponseTransformer_test_df)
    c_dict = {
        "c" + suffix: b_dict["b" + suffix]
        for suffix in ["", "_blue", "_yellow", "_green"]
    }

    full_dict.update(b_dict)
    full_dict.update(c_dict)

    return full_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_mean():
    return_dict = {
        "b": (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6,
        "b_blue": (1.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6,
        "b_yellow": (0.0 + 0.0 + 1.0 + 1.0 + 0.0 + 0.0) / 6,
        "b_green": (0.0 + 0.0 + 0.0 + 0.0 + 1.0 + 1.0) / 6,
    }

    for key, value in return_dict.items():
        return_dict[key] = np.float32(value)

    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_median():
    return_dict = {
        "b": 3.0,
        "b_blue": 0.0,
        "b_yellow": 0.0,
        "b_green": 0.0,
    }

    for key, value in return_dict.items():
        return_dict[key] = np.float32(value)
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_highest():
    return_dict = {
        "b": 6.0,
        "b_blue": 1.0,
        "b_yellow": 1.0,
        "b_green": 1.0,
    }
    for key, value in return_dict.items():
        return_dict[key] = np.float32(value)
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_lowest():
    return_dict = {
        "b": 1.0,
        "b_blue": 0.0,
        "b_yellow": 0.0,
        "b_green": 0.0,
    }

    for key, value in return_dict.items():
        return_dict[key] = np.float32(value)
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_arbitrary():
    return_dict = {
        "b": 22.0,
        "b_blue": 22.0,
        "b_yellow": 22.0,
        "b_green": 22.0,
    }
    for key, value in return_dict.items():
        return_dict[key] = np.float32(value)
    return return_dict


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for MeanResponseTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    # overload inherited arg tests that have been replaced by beartype

    def test_weight_arg_errors(self):
        pass

    def test_prior_not_positive_int_error(self):
        """Test that an exception is raised if prior is not a positive int."""
        with pytest.raises(BeartypeCallHintParamViolation):
            MeanResponseTransformer(prior=-1)


class TestPriorRegularisation:
    "tests for _prior_regularisation method."

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_output1(self, library):
        "Test output of method."
        x = MeanResponseTransformer(columns="a", prior=3)

        df_dict = {"a": [1, 2]}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        y = nw.new_series(name="y", values=[2, 3], backend=library)

        x.fit(X=df, y=y)

        expected1 = (1 + 3 * 2.5) / (1 + 3)

        expected2 = (2 + 3 * 2.5) / (2 + 3)

        expected = {"a": expected1, "b": expected2}

        weights_column = "weights"
        response_column = "means"
        column = "column"
        group_means_and_weights_dict = {
            column: ["a", "b"],
            response_column: [1, 2],
            weights_column: [1, 2],
        }

        group_means_and_weights_df = dataframe_init_dispatch(
            dataframe_dict=group_means_and_weights_dict,
            library="pandas",
        )

        output = x._prior_regularisation(
            group_means_and_weights=group_means_and_weights_df,
            column=column,
            weights_column=weights_column,
            response_column=response_column,
        )

        assert output == expected, (
            f"output of _prior_regularisation not as expected, expected {expected} but got {output}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_output2(self, library):
        "Test output of method - for category dtypes"
        x = MeanResponseTransformer(columns="a", prior=0)

        df_dict = {"a": ["a", "b"]}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("a").cast(nw.Categorical))
        df = nw.to_native(df)

        y = nw.new_series(name="y", values=[2, 3], backend=library)

        x.fit(X=df, y=y)

        expected1 = (1) / (1)

        expected2 = (2) / (2)

        expected = {"a": expected1, "b": expected2}

        weights_column = "weights"
        response_column = "means"
        column = "column"
        group_means_and_weights_dict = {
            column: ["a", "b"],
            response_column: [1, 2],
            weights_column: [1, 2],
        }

        group_means_and_weights_df = dataframe_init_dispatch(
            dataframe_dict=group_means_and_weights_dict,
            library="pandas",
        )

        output = x._prior_regularisation(
            group_means_and_weights=group_means_and_weights_df,
            column=column,
            weights_column=weights_column,
            response_column=response_column,
        )

        assert output == expected, (
            f"output of _prior_regularisation not as expected, expected {expected} but got {output}"
        )

    @pytest.mark.parametrize("library", ["pandas"])
    def test_output3(self, library):
        "Test output of method - for pandas object dtype"
        x = MeanResponseTransformer(columns="a", prior=0)

        df_dict = {"a": ["a", "b"]}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df["a"] = df["a"].astype("object")

        y = nw.new_series(name="y", values=[2, 3], backend=library)

        x.fit(X=df, y=y)

        expected1 = (1) / (1)

        expected2 = (2) / (2)

        expected = {"a": expected1, "b": expected2}

        weights_column = "weights"
        response_column = "means"
        column = "column"
        group_means_and_weights_dict = {
            column: ["a", "b"],
            response_column: [1, 2],
            weights_column: [1, 2],
        }

        group_means_and_weights_df = dataframe_init_dispatch(
            dataframe_dict=group_means_and_weights_dict,
            library="pandas",
        )

        output = x._prior_regularisation(
            group_means_and_weights=group_means_and_weights_df,
            column=column,
            weights_column=weights_column,
            response_column=response_column,
        )

        assert output == expected, (
            f"output of _prior_regularisation not as expected, expected {expected} but got {output}"
        )


class TestFit(GenericFitTests, WeightColumnFitMixinTests, DummyWeightColumnMixinTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_weights_column_missing_error(self, library):
        """Test that an exception is raised if weights_column is specified but not present in data for fit."""
        df = create_MeanResponseTransformer_test_df(library=library)

        x = MeanResponseTransformer(weights_column="z", columns=["b", "d", "f"])

        with pytest.raises(
            ValueError,
            match=r"weight col \(z\) is not present in columns of data",
        ):
            x.fit(
                df,
                df["a"],
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "mean"),
            ("all", "multi_level_response", 32),
            (["yellow", "blue"], "multi_level_response", "max"),
        ],
    )
    def test_response_column_nulls_error(
        self,
        level,
        target_column,
        unseen_level_handling,
        library,
    ):
        """Test that an exception is raised if nulls are present in response_column."""
        df = create_MeanResponseTransformer_test_df(library=library)

        df = nw.from_native(df)
        df = df.with_columns(
            nw.new_series(
                name=target_column,
                values=[*[1.0] * (len(df) - 1), None],
                backend=library,
            ),
        )
        df = nw.to_native(df)

        x = MeanResponseTransformer(
            columns=["b"],
            level=level,
            unseen_level_handling=unseen_level_handling,
        )

        with pytest.raises(
            ValueError,
            match="MeanResponseTransformer: y has 1 null values",
        ):
            x.fit(df, df[target_column])

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "mean"),
            (None, "a", "min"),
        ],
    )
    def test_correct_mappings_stored_numeric_response(
        self,
        learnt_mapping_dict,
        level,
        target_column,
        unseen_level_handling,
        library,
    ):
        "Test that the mapping dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df(library=library)
        columns = ["b", "c"]
        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        assert x.columns == columns, "Columns attribute changed in fit"

        for column in x.columns:
            actual = x.mappings[column]
            expected = learnt_mapping_dict[column]
            assert actual == expected

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (["blue"], "multi_level_response", "median"),
            ("all", "multi_level_response", 32),
            (["yellow", "blue"], "multi_level_response", "max"),
        ],
    )
    def test_correct_mappings_stored_categorical_response(
        self,
        learnt_mapping_dict,
        level,
        target_column,
        unseen_level_handling,
        library,
    ):
        "Test that the mapping dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df(library=library)
        columns = ["b", "c"]
        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        df = nw.from_native(df)

        if level == "all":
            expected_created_cols = {
                prefix + "_" + suffix
                for prefix, suffix in product(
                    columns,
                    df[target_column].unique().to_list(),
                )
            }

        else:
            expected_created_cols = {
                prefix + "_" + suffix for prefix, suffix in product(columns, level)
            }
        assert set(x.encoded_columns) == expected_created_cols, (
            "Stored encoded columns are not as expected"
        )

        for column in x.encoded_columns:
            actual = x.mappings[column]
            expected = learnt_mapping_dict[column]
            assert actual == expected

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "mean"),
            (None, "a", "median"),
            (None, "a", "min"),
            (None, "a", "max"),
            (None, "a", 22.0),
            ("all", "multi_level_response", "mean"),
            (["yellow", "blue"], "multi_level_response", "mean"),
        ],
    )
    def test_correct_unseen_levels_encoding_dict_stored(
        self,
        learnt_unseen_levels_encoding_dict_mean,
        learnt_unseen_levels_encoding_dict_median,
        learnt_unseen_levels_encoding_dict_lowest,
        learnt_unseen_levels_encoding_dict_highest,
        learnt_unseen_levels_encoding_dict_arbitrary,
        level,
        target_column,
        unseen_level_handling,
        library,
    ):
        "Test that the unseen_levels_encoding_dict dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df(library=library)
        x = MeanResponseTransformer(
            columns=["b"],
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        if level:
            if level == "all":
                assert set(x.unseen_levels_encoding_dict.keys()) == {
                    "b_blue",
                    "b_yellow",
                    "b_green",
                }, "Stored unseen_levels_encoding_dict keys are not as expected"

            else:
                assert set(x.unseen_levels_encoding_dict.keys()) == {
                    "b_blue",
                    "b_yellow",
                }, "Stored unseen_levels_encoding_dict keys are not as expected"

            for column in x.unseen_levels_encoding_dict:
                actual = x.unseen_levels_encoding_dict[column]
                expected = learnt_unseen_levels_encoding_dict_mean[column]
                assert actual == expected

        else:
            assert x.unseen_levels_encoding_dict.keys() == {
                "b",
            }, "Stored unseen_levels_encoding_dict key is not as expected"

            if unseen_level_handling == "mean":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_mean[column]
                    assert actual == expected

            elif unseen_level_handling == "median":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_median[column]
                    assert actual == expected

            elif unseen_level_handling == "min":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_lowest[column]
                    assert actual == expected

            elif unseen_level_handling == "max":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_highest[column]
                    assert actual == expected

            else:
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_arbitrary[column]
                    assert actual == expected

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_missing_categories_ignored(self, library):
        "test that where a categorical column has missing levels, these do not make it into the encoding dict"

        df = create_MeanResponseTransformer_test_df(library=library)
        unobserved_value = "a"
        df = nw.from_native(df)
        df = df.filter(nw.col("c") == unobserved_value)
        df = nw.to_native(df)

        target_column = "e"
        x = MeanResponseTransformer(
            columns=["c"],
        )
        x.fit(df, df[target_column])

        assert unobserved_value not in x.mappings, (
            "MeanResponseTransformer should ignore unobserved levels"
        )


class TestFitBinaryResponse(GenericFitTests, WeightColumnFitMixinTests):
    """Tests for MeanResponseTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        (
            "columns",
            "weights_values",
            "prior",
            "expected_mappings",
            "expected_mean",
        ),
        [
            # no prior, no weight
            (
                ["b", "d", "f"],
                [1, 1, 1, 1, 1, 1],
                0,
                {
                    "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
                    "d": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0},
                    "f": {False: 2.0, True: 5.0},
                },
                np.float64(3.5),
            ),
            # no weight, prior
            (
                ["b", "d", "f"],
                [1, 1, 1, 1, 1, 1],
                5,
                {
                    "b": {
                        "a": 37 / 12,
                        "b": 13 / 4,
                        "c": 41 / 12,
                        "d": 43 / 12,
                        "e": 15 / 4,
                        "f": 47 / 12,
                    },
                    "d": {
                        1: 37 / 12,
                        2: 13 / 4,
                        3: 41 / 12,
                        4: 43 / 12,
                        5: 15 / 4,
                        6: 47 / 12,
                    },
                    "f": {False: 47 / 16, True: 65 / 16},
                },
                np.float64(3.5),
            ),
            # weight, no prior
            (
                ["b", "d", "f"],
                [1, 2, 3, 4, 5, 6],
                0,
                {
                    "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
                    "d": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0},
                    "f": {False: 14 / 6, True: 77 / 15},
                },
                np.float64(13 / 3),
            ),
            # prior and weight
            (
                ["d", "f"],
                [1, 1, 1, 2, 2, 2],
                5,
                {
                    "d": {1: 7 / 2, 2: 11 / 3, 3: 23 / 6, 4: 4.0, 5: 30 / 7, 6: 32 / 7},
                    "f": {False: 13 / 4, True: 50 / 11},
                },
                np.float64(4.0),
            ),
        ],
    )
    def test_learnt_values(
        self,
        library,
        columns,
        weights_values,
        prior,
        expected_mappings,
        expected_mean,
    ):
        """Test that the mean response values learnt during fit are expected."""
        df = create_MeanResponseTransformer_test_df(library=library)

        df = nw.from_native(df)

        weights_column = "weights_column"
        df = df.with_columns(
            nw.new_series(name=weights_column, values=weights_values, backend=library),
        ).to_native()

        x = MeanResponseTransformer(columns=columns, prior=prior)

        x.mappings = {}

        x._fit_binary_response(
            df,
            x.columns,
            weights_column=weights_column,
            response_column="a",
        )

        for key in expected_mappings:
            for value in expected_mappings[key]:
                expected_mappings[key][value] = x.cast_method(
                    expected_mappings[key][value],
                )

        assert x.global_mean == expected_mean, (
            f"global mean not learnt as expected, expected {expected_mean} but got {x.global_mean}"
        )

        assert x.mappings == expected_mappings, (
            f"mappings not learnt as expected, expected {expected_mappings} but got {x.mappings}"
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("prior", (1, 3, 5, 7, 9, 11, 100))
    def test_prior_logic(self, prior, library):
        "Test that for prior>0 encodings are closer to global mean than for prior=0."
        df = create_MeanResponseTransformer_test_df(library=library)

        df = nw.from_native(df)
        weights_column = "weights_column"
        native_backend = nw.get_native_namespace(df)
        df = df.with_columns(
            nw.new_series(
                name=weights_column,
                values=[1, 1, 1, 2, 2, 2],
                backend=native_backend.__name__,
            ),
        )

        x_prior = MeanResponseTransformer(
            columns=["d", "f"],
            prior=prior,
            weights_column=weights_column,
        )

        x_no_prior = MeanResponseTransformer(
            columns=["d", "f"],
            prior=0,
            weights_column=weights_column,
        )

        x_prior.mappings = {}
        x_no_prior.mappings = {}

        x_prior._fit_binary_response(
            df,
            x_prior.columns,
            response_column="a",
            weights_column=weights_column,
        )

        x_no_prior._fit_binary_response(
            df,
            x_no_prior.columns,
            weights_column=weights_column,
            response_column="a",
        )

        prior_mappings = x_prior.mappings

        no_prior_mappings = x_no_prior.mappings

        global_mean = x_prior.global_mean

        assert global_mean == x_no_prior.global_mean, (
            "global means for transformers with/without priors should match"
        )

        for col in prior_mappings:
            for value in prior_mappings[col]:
                prior_encoding = prior_mappings[col][value]
                no_prior_encoding = no_prior_mappings[col][value]

                prior_mean_dist = np.abs(prior_encoding - global_mean)
                no_prior_mean_dist = np.abs(no_prior_encoding - global_mean)

                assert prior_mean_dist <= no_prior_mean_dist, (
                    "encodings using priors should be closer to the global mean than without"
                )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("low_weight", "high_weight"),
        ((1, 2), (2, 3), (3, 4), (10, 20)),
    )
    def test_prior_logic_for_weights(self, low_weight, high_weight, library):
        "Test that for fixed prior a group with lower weight is moved closer to the global mean than one with higher weight."
        df = create_MeanResponseTransformer_test_df(library=library)

        df = nw.from_native(df)
        weights_column = "weights_column"
        native_backend = nw.get_native_namespace(df)

        # column f looks like [False, False, False, True, True, True]
        df = df.with_columns(
            nw.new_series(
                name=weights_column,
                values=[
                    low_weight,
                    low_weight,
                    low_weight,
                    high_weight,
                    high_weight,
                    high_weight,
                ],
                backend=native_backend.__name__,
            ),
        )

        x_prior = MeanResponseTransformer(
            columns=["f"],
            prior=5,
            weights_column=weights_column,
        )

        x_no_prior = MeanResponseTransformer(
            columns=["f"],
            prior=0,
            weights_column=weights_column,
        )

        x_prior.mappings = {}
        x_no_prior.mappings = {}

        x_prior._fit_binary_response(
            df,
            x_prior.columns,
            weights_column=weights_column,
            response_column="a",
        )

        x_no_prior._fit_binary_response(
            df,
            x_no_prior.columns,
            weights_column=weights_column,
            response_column="a",
        )

        prior_mappings = x_prior.mappings

        no_prior_mappings = x_no_prior.mappings

        global_mean = x_prior.global_mean

        assert global_mean == x_no_prior.global_mean, (
            "global means for transformers with/without priors should match"
        )

        low_weight_prior_encoding = prior_mappings["f"][False]
        high_weight_prior_encoding = prior_mappings["f"][True]

        low_weight_no_prior_encoding = no_prior_mappings["f"][False]
        high_weight_no_prior_encoding = no_prior_mappings["f"][True]

        low_weight_prior_mean_dist = np.abs(low_weight_prior_encoding - global_mean)
        high_weight_prior_mean_dist = np.abs(high_weight_prior_encoding - global_mean)

        low_weight_no_prior_mean_dist = np.abs(
            low_weight_no_prior_encoding - global_mean,
        )
        high_weight_no_prior_mean_dist = np.abs(
            high_weight_no_prior_encoding - global_mean,
        )

        # check low weight group has been moved further towards mean than high weight group by prior, i.e
        # that the distance remaining is a smaller proportion of the no prior distance
        low_ratio = low_weight_prior_mean_dist / low_weight_no_prior_mean_dist
        high_ratio = high_weight_prior_mean_dist / high_weight_no_prior_mean_dist
        assert low_ratio <= high_ratio, (
            "encodings for categories with lower weights should be moved closer to the global mean than those with higher weights, for fixed prior"
        )


def expected_df_1(library="pandas"):
    """Expected output for single level response."""

    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1, 2, 3, 4, 5, 6],
        "c": ["a", "b", "c", "d", "e", "f"],
        "d": [1, 2, 3, 4, 5, 6],
        "e": [1, 2, 3, 4, 5, 6],
        "f": [2, 2, 2, 5, 5, 5],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_2(library="pandas"):
    """Expected output for response with level = blue."""

    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "c": ["a", "b", "c", "d", "e", "f"],
        "d": [1, 2, 3, 4, 5, 6],
        "e": [1, 2, 3, 4, 5, 6],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
        ],
        "b_blue": [1, 1, 0, 0, 0, 0],
        "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_3(library="pandas"):
    """Expected output for response with level = 'all'."""

    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "c": ["a", "b", "c", "d", "e", "f"],
        "d": [1, 2, 3, 4, 5, 6],
        "e": [1, 2, 3, 4, 5, 6],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
        ],
        "b_blue": [1, 1, 0, 0, 0, 0],
        "f_blue": [2 / 3, 2 / 3, 2 / 3, 0.0, 0.0, 0.0],
        "b_green": [0, 0, 0, 0, 1, 1],
        "f_green": [0.0, 0.0, 0.0, 2 / 3, 2 / 3, 2 / 3],
        "b_yellow": [0, 0, 1, 1, 0, 0],
        "f_yellow": [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_4(library="pandas"):
    """Expected output for transform on dataframe with single level response and unseen levels,
    where unseen_level_handling = 'mean'.
    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_5(library="pandas"):
    """Expected output for transform on dataframe with single level response and unseen levels,
    where unseen_level_handling = 'median'.
    """

    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_6(library="pandas"):
    """Expected output for transform on dataframe with single level response and unseen levels,
    where unseen_level_handling = 'min'.
    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_7(library="pandas"):
    """Expected output for transform on dataframe with single level response and unseen levels,
    where unseen_level_handling = 'max'.
    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_8(library="pandas"):
    """Expected output for transform on dataframe with single level response and unseen levels,
    where unseen_level_handling set to arbitrary int/float value.
    """
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 21.6, 21.6],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 21.6, 21.6],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_9(library="pandas"):
    """Expected output for transform on dataframe with multi-level response with level = blue and unseen levels in data."""
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1, 2, 3, 4, 5, 6, 7, 8],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
        "b_blue": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2 / 6, 2 / 6],
        "f_blue": [2 / 3, 2 / 3, 2 / 3, 0.0, 0.0, 0.0, 0, 2 / 3],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


def expected_df_10(library="pandas"):
    """Expected output for transform on dataframe with multi-level response with level = "all" and unseen levels in data."""
    df_dict = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
        "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "d": [1, 2, 3, 4, 5, 6, 7, 8],
        "e": [1, 2, 3, 4, 5, 6, 7, 8],
        "multi_level_response": [
            "blue",
            "blue",
            "yellow",
            "yellow",
            "green",
            "green",
            "yellow",
            "blue",
        ],
        "b_blue": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0, 0, 2 / 3],
        "b_yellow": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "f_yellow": [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
        "b_green": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "f_green": [0.0, 0.0, 0.0, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 0],
    }

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
    df = nw.from_native(df)
    df = df.with_columns(nw.col("c").cast(nw.Categorical))

    return df.to_native()


class TestTransform(GenericTransformTests):
    """Tests for MeanResponseTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        (
            "columns",
            "level",
            "mappings",
            "column_to_encoded_columns",
            "expected_getter",
        ),
        [
            # binary response
            (
                ["b", "d", "f"],
                None,
                {
                    "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                    "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                    "f": {False: 2, True: 5},
                },
                {"b": ["b"], "d": ["d"], "f": ["f"]},
                expected_df_1,
            ),
            # multi level, single level
            (
                ["b", "f"],
                ["blue"],
                {
                    "b_blue": {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0, "f": 0},
                    "f_blue": {False: 2 / 3, True: 0.0},
                },
                {"b": ["b_blue"], "f": ["f_blue"]},
                expected_df_2,
            ),
            # multi level, all levels
            (
                ["b", "f"],
                "all",
                {
                    "b_blue": {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0, "f": 0},
                    "b_yellow": {"a": 0, "b": 0, "c": 1, "d": 1, "e": 0, "f": 0},
                    "b_green": {"a": 0, "b": 0, "c": 0, "d": 0, "e": 1, "f": 1},
                    "f_blue": {False: 2 / 3, True: 0.0},
                    "f_yellow": {False: 1 / 3, True: 1 / 3},
                    "f_green": {False: 0.0, True: 2 / 3},
                },
                {
                    "b": ["b_blue", "b_yellow", "b_green"],
                    "f": ["f_blue", "f_yellow", "f_green"],
                },
                expected_df_3,
            ),
        ],
    )
    def test_expected_outputs(
        self,
        library,
        columns,
        level,
        mappings,
        column_to_encoded_columns,
        expected_getter,
    ):
        """Test that the output is expected from transform with various parametrized setups."""

        df = create_MeanResponseTransformer_test_df(library=library)
        expected = expected_getter(library=library)

        x = MeanResponseTransformer(columns=columns, level=level)

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.mappings = mappings

        # mimic fit logics use of BaseMappingTransformer
        # use BaseMappingTransformer init to process args
        # extract null_mappings from mappings etc
        base_mapping_transformer = BaseMappingTransformer(
            mappings=mappings,
        )

        x.mappings = base_mapping_transformer.mappings
        x.mappings_from_null = base_mapping_transformer.mappings_from_null

        x.column_to_encoded_columns = column_to_encoded_columns
        x.response_levels = level
        x.encoded_columns = list(x.mappings.keys())
        x.return_dtypes = dict.fromkeys(x.encoded_columns, "Float32")

        df_transformed = x.transform(df)

        expected = nw.from_native(expected)
        for col in x.encoded_columns:
            expected = expected.with_columns(
                nw.col(col).cast(getattr(nw, x.return_type)),
            )
        expected = nw.to_native(expected)

        column_order = nw.from_native(df_transformed).columns

        assert_frame_equal_dispatch(
            df_transformed,
            expected[column_order],
        )

        # also test single row transform
        df = nw.from_native(df)
        expected = nw.from_native(expected)

        for i in range(len(df)):
            df_transformed_row = x.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            column_order = nw.from_native(df_transformed_row).columns

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row[column_order],
            )

    # NOTE - this currently is more of a Fit test imo, but will leave in place for now
    # as does also test Transform
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("columns", "target", "unseen_level_handling", "level", "expected_getter"),
        [
            (["b", "d", "f"], "a", "mean", None, expected_df_4),
            (["b", "d", "f"], "a", "median", None, expected_df_5),
            (["b", "d", "f"], "a", "min", None, expected_df_6),
            (["b", "d", "f"], "a", "max", None, expected_df_7),
            (["b", "d", "f"], "a", 21.6, None, expected_df_8),
            (["b", "f"], "multi_level_response", "mean", ["blue"], expected_df_9),
            (["b", "f"], "multi_level_response", "max", "all", expected_df_10),
        ],
    )
    def test_expected_outputs_with_unseen_level_handling(
        self,
        library,
        columns,
        target,
        unseen_level_handling,
        level,
        expected_getter,
    ):
        """Test that the output is expected from transform with various configs and unseen level handling"""

        df = create_MeanResponseTransformer_test_df_unseen_levels(library=library)
        expected = expected_getter(library=library)

        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling=unseen_level_handling,
        )

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])

        df_transformed = x.transform(df)

        expected = nw.from_native(expected)
        for col in x.encoded_columns:
            expected = expected.with_columns(
                nw.col(col).cast(getattr(nw, x.return_type)),
            )

        column_order = nw.from_native(df_transformed).columns

        assert_frame_equal_dispatch(
            df_transformed,
            expected[column_order].to_native(),
        )

        # also test single row transform
        df = nw.from_native(df)
        expected = nw.from_native(expected)

        for i in range(len(df)):
            df_transformed_row = x.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            column_order = nw.from_native(df_transformed_row).columns

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row[column_order],
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_nulls_introduced_in_transform_error(self, library):
        """Test that transform will raise an error if nulls are introduced."""
        df = create_MeanResponseTransformer_test_df(library=library)

        x = MeanResponseTransformer(columns=["b", "d", "f"])

        x.fit(df, df["a"])

        df = nw.from_native(df)
        df = df.with_columns(nw.lit("z").alias("b"))
        df = nw.to_native(df)

        with pytest.raises(
            ValueError,
            match="MeanResponseTransformer: nulls would be introduced into columns b from levels not present in mapping",
        ):
            x.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "prior, level, target, unseen_level_handling",
        [
            (5, "all", "c", "mean"),
            (100, ["a", "b"], "c", "min"),
            (1, None, "a", "max"),
            (0, None, "a", "median"),
        ],
    )
    def test_return_type_can_be_changed(
        self,
        prior,
        level,
        target,
        unseen_level_handling,
        library,
    ):
        "Test that output return types are controlled by return_type param, this defaults to float32 so test float64 here"

        df = create_MeanResponseTransformer_test_df(library=library)

        columns = ["b", "d", "f"]
        x = MeanResponseTransformer(
            columns=columns,
            return_type="Float64",
            prior=prior,
            unseen_level_handling=unseen_level_handling,
            level=level,
        )

        x.fit(df, df[target])

        output_df = nw.from_native(x.transform(df))

        if target == "c":
            actual_levels = df[target].unique().to_list() if level == "all" else level
            expected_created_cols = [
                prefix + "_" + suffix
                for prefix, suffix in product(columns, actual_levels)
            ]

        else:
            expected_created_cols = columns

        for col in expected_created_cols:
            expected_type = x.return_type
            actual_type = str(output_df[col].dtype)
            assert actual_type == expected_type, (
                f"{x.classname} should output columns with type determine by the return_type param, expected {expected_type} but got {actual_type}"
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified(self, library):
        """Test that the mappings from fit are not changed in transform."""
        df = create_MeanResponseTransformer_test_df(library=library)

        x = MeanResponseTransformer(columns="b")

        x.fit(df, df["a"])

        x2 = MeanResponseTransformer(columns="b")

        x2.fit(df, df["a"])

        x2.transform(df)

        assert x.mappings == x2.mappings, (
            "Mean response values not changed in transform"
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"
