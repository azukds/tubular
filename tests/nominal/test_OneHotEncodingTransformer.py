import narwhals as nw
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from test_BaseNominalTransformer import GenericNominalTransformTests

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericFitTests,
    GenericTransformTests,
    SeparatorInitMixintests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.nominal import OneHotEncodingTransformer


class TestInit(
    SeparatorInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"

    # Tests for wanted_values parameter

    @pytest.mark.parametrize(
        "values",
        ["a", ["a", "b"], 123, True],
    )
    def test_wanted_values_is_dict(self, values, minimal_attribute_dict):
        args = minimal_attribute_dict[self.transformer_name]
        args["wanted_values"] = values

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneHotEncodingTransformer(**args)

    @pytest.mark.parametrize(
        "values",
        [
            {1: ["a", "b"]},
            {True: ["a"]},
            {("a",): ["b", "c"]},
        ],
    )
    def test_wanted_values_key_is_str(self, values, minimal_attribute_dict):
        args = minimal_attribute_dict[self.transformer_name]
        args["wanted_values"] = values

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneHotEncodingTransformer(**args)

    @pytest.mark.parametrize(
        ("values", "columns"),
        [({"a": ["b"]}, ["b"]), ({"a": ["b"], "z": ["x", "y"]}, ["a", "b"])],
    )
    def test_wanted_values_not_matching_columns_error(
        self, values, columns, minimal_attribute_dict
    ):
        args = minimal_attribute_dict[self.transformer_name]
        args["wanted_values"] = values
        args["columns"] = columns

        msg = f"{self.transformer_name}: keys of wanted values should match provided columns"
        with pytest.raises(ValueError, match=msg):
            OneHotEncodingTransformer(**args)

    @pytest.mark.parametrize(
        "values",
        [
            {"a": "b"},
            {"a": ("a", "b")},
            {"a": True},
            {"a": 123},
        ],
    )
    def test_wanted_values_value_is_list(self, values, minimal_attribute_dict):
        args = minimal_attribute_dict[self.transformer_name]
        args["wanted_values"] = values

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneHotEncodingTransformer(**args)

    @pytest.mark.parametrize(
        "values",
        [
            {"a": ["b", 123]},
            {"a": ["b", True]},
            {"a": ["b", None]},
            {"a": ["b", ["a", "b"]]},
        ],
    )
    def test_wanted_values_entries_are_str(self, values, minimal_attribute_dict):
        args = minimal_attribute_dict[self.transformer_name]
        args["wanted_values"] = values

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneHotEncodingTransformer(**args)

    # overload this test until we beartype separator mixin
    @pytest.mark.parametrize(
        "separator",
        [1, True, {"a": 1}, [1, 2], None, np.inf, np.nan],
    )
    def test_separator_type_error(
        self,
        separator,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if any type other than str passed to separator"""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["separator"] = separator

        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nulls_handled_successfully_error(self, library, lazy):
        """Test that fit does not error if columns contain nulls"""
        df = d.create_df_2(library=library)

        transformer = OneHotEncodingTransformer(columns=["b", "c"])

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_fields_with_over_100_levels_error(self, library, lazy):
        """Test that OneHotEncodingTransformer.fit on fields with more than 100 levels raises error."""
        df_dict = {"a": [1] * 101, "b": list(range(101))}

        df = dataframe_init_dispatch(library=library, dataframe_dict=df_dict)

        transformer = OneHotEncodingTransformer(columns=["a", "b"])

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        with pytest.raises(
            ValueError,
            match="OneHotEncodingTransformer: column b has over 100 unique values - consider another type of encoding",
        ):
            transformer.fit(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_fields_with_over_100_levels_do_not_error_with_wanted_levels(
        self, library, lazy
    ):
        """Test that OneHotEncodingTransformer.fit bypasses too many levels error if wanted_levels is provided"""
        df_dict = {"a": ["1"] * 101, "b": list(range(101))}

        df = dataframe_init_dispatch(library=library, dataframe_dict=df_dict)

        transformer = OneHotEncodingTransformer(
            columns=["a"], wanted_values={"a": ["1"]}
        )

        if _check_if_skip_test(transformer, df, lazy=lazy):
            return

        transformer.fit(_convert_to_lazy(df, lazy=lazy))


class TestTransform(
    DropOriginalTransformMixinTests,
    GenericNominalTransformTests,
    GenericTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"

    def create_OneHotEncoderTransformer_test_df_1(self, library="pandas"):
        """Create DataFrame to test OneHotEncoderTransformer

        binary columns are representative of transformed output of column b

        Parameters
        ----------
        library : str
            Whether to return polars of pandas df

        """

        df_dict = {
            "a": [4, 2, 2, 1, 3],
            "b": ["x", "z", "y", "x", "x"],
            "c": ["c", "a", "a", "c", "b"],
            "b_x": [1.0, 0.0, 0.0, 1.0, 1.0],
            "b_y": [0.0, 0.0, 1.0, 0.0, 0.0],
            "b_z": [0.0, 1.0, 0.0, 0.0, 0.0],
        }

        df = dataframe_init_dispatch(library=library, dataframe_dict=df_dict)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return df.to_native()

    def create_OneHotEncoderTransformer_test_df_2(self, library="pandas"):
        """Create DataFrame to test OneHotEncoderTransformer

        binary columns are representative of transformed output of all columns

        Parameters
        ----------
        library : str
            Whether to return polars of pandas df

        """

        df_dict = {
            "a": [1, 5, 2, 3, 3],
            "b": ["w", "w", "z", "y", "x"],
            "c": ["a", "a", "c", "b", "a"],
            "a_1": [1.0, 0.0, 0.0, 0.0, 0.0],
            "a_2": [0.0, 0.0, 1.0, 0.0, 0.0],
            "a_3": [0.0, 0.0, 0.0, 1.0, 1.0],
            "a_4": [0.0, 0.0, 0.0, 0.0, 0.0],
            "b_x": [0.0, 0.0, 0.0, 0.0, 1.0],
            "b_y": [0.0, 0.0, 0.0, 1.0, 0.0],
            "b_z": [0.0, 0.0, 1.0, 0.0, 0.0],
            "c_a": [1.0, 1.0, 0.0, 0.0, 1.0],
            "c_b": [0.0, 0.0, 0.0, 1.0, 0.0],
            "c_c": [0.0, 0.0, 1.0, 0.0, 0.0],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return df.to_native()

    def test_non_mappable_rows_exception_raised(self):
        """Test inherited from GenericBaseNominalTransformerTests needs to be overwritten,
        inherited test tests the mapping attribute, which OHE transformer doesn't have.
        """

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_expected_output(self, library, from_json, lazy):
        """Test that OneHotEncodingTransformer.transform encodes the feature correctly.

        Also tests that OneHotEncodingTransformer.transform does not modify unrelated columns.
        """
        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        columns = ["b"]
        df_train = d.create_df_7(library=library)
        df_train = nw.from_native(df_train)

        df_test = df_train.clone()
        expected = self.create_OneHotEncoderTransformer_test_df_1(library=library)

        transformer = OneHotEncodingTransformer(columns=columns)

        if _check_if_skip_test(transformer, df_train, lazy=lazy, from_json=from_json):
            return

        transformer.fit(_convert_to_lazy(df_train, lazy=lazy))

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(
            _convert_to_lazy(df_test.to_native(), lazy=lazy)
        )

        expected = nw.from_native(expected)
        for col in [
            column + f"_{value}"
            for column in columns
            for value in df_train.select(nw.col(column).unique())
            .get_column(column)
            .to_list()
        ]:
            expected = expected.with_columns(nw.col(col).cast(nw.Boolean))

        assert_frame_equal_dispatch(
            expected.to_native(), _collect_frame(df_transformed, lazy=lazy)
        )

        # also test single row transform
        for i in range(len(df_test)):
            df_transformed_row = transformer.transform(df_test[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_categories_not_modified(self, library, from_json, lazy):
        """Test that the categories from fit are not changed in transform."""
        df_train = d.create_df_1(library=library)
        df_test = d.create_df_7(library=library)

        transformer = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)
        transformer2 = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)

        if _check_if_skip_test(transformer, df_train, lazy=lazy, from_json=from_json):
            return

        transformer.fit(_convert_to_lazy(df_train, lazy=lazy))
        transformer2.fit(_convert_to_lazy(df_train, lazy=lazy))

        transformer = _handle_from_json(transformer, from_json)

        transformer.transform(_convert_to_lazy(df_test, lazy=lazy))

        assert transformer2.categories_ == transformer.categories_, (
            f"categories_ modified during transform, pre transform had {transformer2.categories_} but post transform has {transformer.categories_}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_renaming_feature_works_as_expected(self, library, from_json, lazy):
        """Test OneHotEncodingTransformer.transform() is renaming features correctly."""
        df = d.create_df_7(library=library)
        df = df[["b", "c"]]

        transformer = OneHotEncodingTransformer(
            columns=["b", "c"],
            separator="|",
            drop_original=True,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df)

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        expected_columns = ["b|x", "b|y", "b|z", "c|a", "c|b", "c|c"]

        df_transformed = nw.from_native(df_transformed)
        actual_columns = df_transformed.columns

        assert set(expected_columns) == set(actual_columns), (
            f"renaming columns feature in OneHotEncodingTransformer.transform, expected {expected_columns} but got {actual_columns}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_unseen_categories_encoded_as_all_zeroes(self, library, from_json, lazy):
        """Test OneHotEncodingTransformer.transform encodes unseen categories correctly (all 0s)."""
        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        df_train = d.create_df_7(library=library)

        columns = ["a", "b", "c"]
        transformer = OneHotEncodingTransformer(columns=columns, verbose=False)

        if _check_if_skip_test(transformer, df_train, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df_train)

        transformer = _handle_from_json(transformer, from_json)

        df_test = d.create_df_8(library=library)
        expected = self.create_OneHotEncoderTransformer_test_df_2(library=library)

        df_transformed = transformer.transform(_convert_to_lazy(df_test, lazy=lazy))

        df_train = nw.from_native(df_train)
        expected = nw.from_native(expected)

        for col in [
            column + f"_{value}"
            for column in columns
            for value in df_train.select(nw.col(column).unique())
            .get_column(column)
            .to_list()
        ]:
            expected = expected.with_columns(nw.col(col).cast(nw.Boolean))

        column_order = expected.columns
        assert_frame_equal_dispatch(
            expected.to_native(),
            _collect_frame(df_transformed, lazy=lazy)[column_order],
        )

        # also test single row transform
        df_test = nw.from_native(df_test)
        for i in range(len(df_test)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df_test[[i]].to_native(), lazy=lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy)[column_order],
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_transform_output_with_wanted_values_arg(self, library, from_json, lazy):
        """
        Test to verify OneHotEncodingTransformer.transform zero-filled levels from user-specified "wanted_levels" and encodes only those listed in "wanted_levels".

        """
        df_train = d.create_df_7(library=library)
        df_test = d.create_df_8(library=library)

        transformer = OneHotEncodingTransformer(
            columns=["b"],
            wanted_values={"b": ["v", "x", "z"]},
        )

        if _check_if_skip_test(transformer, df_train, lazy=lazy, from_json=from_json):
            return

        transformer.fit(df_train)

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df_test, lazy=lazy))

        expected_df_dict = {
            "a": [1, 5, 2, 3, 3],
            "b": ["w", "w", "z", "y", "x"],
            "c": ["a", "a", "c", "b", "a"],
            "b_v": [0] * 5,
            "b_x": [0, 0, 0, 0, 1],
            "b_z": [0, 0, 1, 0, 0],
        }
        expected_df = dataframe_init_dispatch(
            library=library,
            dataframe_dict=expected_df_dict,
        )
        expected_df = nw.from_native(expected_df)
        # cast the columns
        boolean_cols = ["b_v", "b_x", "b_z"]
        for col_name in boolean_cols:
            expected_df = expected_df.with_columns(
                nw.col(col_name).cast(nw.Boolean),
            )
        expected_df = expected_df.with_columns(
            nw.col("c").cast(nw.Categorical),
        )

        assert_frame_equal_dispatch(
            _collect_frame(df_transformed, lazy=lazy), expected_df.to_native()
        )
