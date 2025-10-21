import re

import narwhals as nw
import numpy as np
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    DummyWeightColumnMixinTests,
    GenericFitTests,
    GenericInitTests,
    GenericTransformTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.capping import BaseCappingTransformer


class GenericCappingInitTests(WeightColumnInitMixinTests, GenericInitTests):
    """Tests for BaseCappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"

    @pytest.mark.parametrize(
        "non_string, cap_type",
        [(1, "capping_values"), (True, "quantiles")],
    )
    def test_columns_list_element_error(
        self,
        non_string,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements - note columns is
        derived from keys of either capping_values or quantiles ."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = {non_string: {1: 2, 3: 4}}
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: all keys in {cap_type} should be str",
            ),
        ):
            uninitialized_transformers[self.transformer_name](**args)

    def test_capping_values_quantiles_both_none_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if both capping_values and quantiles are passed as None."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = None
        args["capping_values"] = None

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: both capping_values and quantiles are None, either supply capping values in the "
            "capping_values argument or supply quantiles that can be learnt in the fit method",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    def test_capping_values_quantiles_both_specified_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if both capping_values and quantiles are specified."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.1, 0.2]}
        args["capping_values"] = {"a": [1, 2]}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: both capping_values and quantiles are not None, supply one or the other",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("out_range_value", [(-2), (1.2)])
    def test_quantiles_outside_range_error(
        self,
        out_range_value,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if quanties contain values outisde [0, 1] range."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"e": [0.1, 0.9], "f": [out_range_value, None]}
        args["capping_values"] = None

        with pytest.raises(
            ValueError,
            match=rf"{self.transformer_name}: quantile values must be in the range \[0, 1\] but got {out_range_value} for key f",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("cap_type", ["capping_values", "quantiles"])
    def test_capping_info_not_dict_error(
        self,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if capping_values or quantiles are not a dict."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = []
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: {cap_type} should be dict of columns and capping values",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("cap_type", ["capping_values", "quantiles"])
    def test_capping_info_non_list_item_error(
        self,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if capping_values or quantiles has any non list items."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = {"b": (None, 1)}
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            TypeError,
            match=rf"{self.transformer_name}: each item in {cap_type} should be a list, but got \<class 'tuple'\> for key b",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("cap_type", ["capping_values", "quantiles"])
    def test_capping_info_non_length_2_list_item_error(
        self,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if capping_values or quantiles has any non length 2 list items."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = {"b": [None]}
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: each item in {cap_type} should be length 2, but got 1 for key b",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("cap_type", ["capping_values", "quantiles"])
    def test_capping_info_non_numeric_error(
        self,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if capping_values or quantiles contains any non-nulls and non-numeric values."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = {"b": [None, "a"]}
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            TypeError,
            match=rf"{self.transformer_name}: each item in {cap_type} lists must contain numeric values or None, got \<class 'str'\> for key b",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("cap_type", ["capping_values", "quantiles"])
    def test_lower_value_gte_upper_value_error(
        self,
        cap_type,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if for capping_values or quantiles dict[0] >= dict[1]."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args[cap_type] = {"b": [0.8, 0.1]}
        # ensure not both capping_values and quantiles are provided
        if cap_type == "quantiles":
            args["capping_values"] = None

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: lower value is greater than or equal to upper value for key b",
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("value", [(np.nan), (np.inf), (-np.inf)])
    def test_capping_value_nan_inf_error(
        self,
        value,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an exception is raised if capping_values are np.nan or np.inf values."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"b": [0.8, value]}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: item in capping_values lists contains numpy NaN or Inf values",
        ):
            uninitialized_transformers[self.transformer_name](**args)


class GenericCappingFitTests(
    WeightColumnFitMixinTests,
    GenericFitTests,
    DummyWeightColumnMixinTests,
):
    """Tests for BaseCappingTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_quantiles_none_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that a warning is raised if quantiles is None when fit is run."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [2, 5], "b": [-1, 8]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        df = d.create_df_3(library=library)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        with pytest.warns(
            UserWarning,
            match=f"{self.transformer_name}: quantiles not set so no fitting done",
        ):
            transformer.fit(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("values", "sample_weight", "quantiles", "expected_quantiles"),
        # quantiles use linear interpolation, which is manually replicated here where needed
        [
            # interpolation formula is val1+(val2-val1)/(cumweight%2-cumweight%1)*(quantile-cumweight%1)
            (
                [1, 2, 3],
                [1, 2, 1],
                [0.1, 0.5],
                [1, 1 + ((2 - 1) / (0.75 - 0.25)) * (0.5 - 0.25)],
            ),  # lower value is 1 as this is the min in range
            (
                [1, 2, 3],
                None,
                [0.1, 0.5],
                [1, 1 + ((2 - 1) / (2 / 3 - 1 / 3)) * (0.5 - 1 / 3)],
            ),  # lower value is 1 as this is the min in range
            (
                [2, 4, 6, 8],
                [3, 2, 1, 1],
                [None, 0.5],
                [None, 2 + ((4 - 2) / (5 / 7 - 3 / 7)) * (0.5 - 3 / 7)],
            ),
            (
                [2, 4, 6, 8],
                None,
                [None, 0.5],
                [None, 2 + ((4 - 2) / (0.5 - 0.25)) * (0.5 - 0.25)],
            ),
            ([-1, -5, -10, 20, 30], [1, 2, 2, 2, 2], [0.1, None], [-10, None]),
            ([-1, -5, -10, 20, 40], None, [0.1, None], [-10, None]),
        ],
    )
    def test_fit_values(
        self,
        values,
        sample_weight,
        quantiles,
        expected_quantiles,
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

        df_dict = {
            "a": values,
        }
        if sample_weight:
            df_dict["w"] = sample_weight

        else:
            transformer.weights_column = None

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        actuals = transformer.quantile_capping_values["a"]

        actuals_dict = {"lower": None, "upper": None}
        names = actuals_dict.keys()

        # round to 1dp to avoid mismatches due to numerical precision
        for name, value in zip(names, actuals):
            if value:
                actuals_dict[name] = np.round(value, 1)
            else:
                actuals_dict[name] = value

        for name, value in zip(names, expected_quantiles):
            assert actuals_dict[name] == value, (
                f"unexpected replacement values fit, for {name} value expected {value} but got {actuals_dict[name]}"
            )

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

        assert transformer._replacement_values == transformer.quantile_capping_values, (
            f"unexpected value for replacement_values attribute, expected {transformer.quantile_capping_values} but got {transformer.replacement_values_}"
        )


class GenericCappingTransformTests(GenericTransformTests):
    """Tests for BaseCappingTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"

    def expected_df_2(self, library="pandas"):
        """Expected output from test_expected_output_max."""

        df_dict = {
            "a": [2, 2, 3, 4, 5, 6, 7, None],
            "b": ["a", "b", "c", "d", "e", "f", "g", None],
            "c": ["a", "b", "c", "d", "e", "f", "g", None],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").cast(nw.Categorical))

        return df.to_native()

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_non_cap_column_left_untouched(
        self,
        initialized_transformers,
        library,
    ):
        """Test that capping is applied only to specific columns, others remain the same."""

        df = d.create_df_4(library=library)

        expected = self.expected_df_2(library=library)

        transformer = initialized_transformers[self.transformer_name]

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        df_transformed = transformer.transform(df)

        # exclude transformed columns for this test
        columns_to_test = [
            col for col in df_transformed.columns if col not in transformer.columns
        ]

        assert_frame_equal_dispatch(
            df_transformed[columns_to_test],
            expected[columns_to_test],
        )

        # Check outcomes for single rows
        df = nw.from_native(df)
        expected = nw.from_native(expected)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row[columns_to_test],
                df_expected_row[columns_to_test],
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "fit_value",
        ["_replacement_values", "capping_values"],
    )
    def test_learnt_values_not_modified(
        self,
        fit_value,
        initialized_transformers,
        library,
    ):
        """Test that the replacements from fit are not changed in transform."""

        transformer = initialized_transformers[self.transformer_name]

        df = d.create_df_3(library=library)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        learnt_values = getattr(transformer, fit_value)

        transformer.transform(df)

        new_learnt_values = getattr(transformer, fit_value)

        assert learnt_values == new_learnt_values, (
            f"learnt_value {fit_value} changed by transform, expected {learnt_values} but got {new_learnt_values}"
        )

        # Check outcomes for single rows
        df = nw.from_native(df)
        for i in range(len(df)):
            transformer.transform(df[[i]].to_native())

            assert learnt_values == new_learnt_values, (
                f"learnt_value {fit_value} changed by transform, expected {learnt_values} but got {new_learnt_values}"
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_non_numeric_column_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if a column to transform is not numeric."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [1, 2]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        df = d.create_df_5(library=library)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        # convert column to non-numeric
        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df)
        df = df.with_columns(
            nw.new_series(
                name="a",
                values=["a"] * len(df),
                backend=native_backend,
            ),
        )

        with pytest.raises(
            TypeError,
            match=rf"{self.transformer_name}: The following columns are not numeric in X; \['a'\]",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_quantile_capping_values_not_fit_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if capping_values attr has not fit"""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.1, 0.2]}
        args["capping_values"] = None

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        with pytest.raises(
            ValueError,
            match=f"This {self.transformer_name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_quantile_capping_values_empty_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if quantile_capping_values is empty dict"""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.1, 0.2]}
        args["capping_values"] = None

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)
        transformer.quantile_capping_values = {}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: quantile_capping_values attribute is an empty dict - perhaps the fit method has not been run yet",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_capping_values_empty_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if capping_values is empty dict"""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [0.1, 0.2]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)
        transformer.capping_values = {}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: capping_values attribute is an empty dict - perhaps the fit method has not been run yet",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_replacement_values_not_fit_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if replacement values attr has not fit"""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.1, 0.2]}
        args["capping_values"] = None

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        with pytest.raises(
            ValueError,
            match=f"This {self.transformer_name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_replacement_values_dict_empty_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that transform will raise an error if _replacement_values is an empty dict."""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.1, 0.2]}
        args["capping_values"] = None

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        # manually set attribute to get past the capping_values attribute is an empty dict exception
        transformer.quantile_capping_values = {"a": [1, 4]}
        transformer._replacement_values = {}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: _replacement_values attribute is an empty dict - perhaps the fit method has not been run yet",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_fixed_attributes_unchanged_from_transform(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
    ):
        """Test that attributes are unchanged after transform is run."""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": [0.2, 1], "b": [0, 1]}
        args["capping_values"] = None

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # if transformer is not polars compatible, skip polars test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        transformer2 = uninitialized_transformers[self.transformer_name](**args)

        transformer2.fit(df)

        transformer2.transform(df)

        assert transformer.weights_column == transformer2.weights_column, (
            "weights_column attribute modified in transform"
        )
        assert transformer.quantiles == transformer2.quantiles, (
            "quantiles attribute modified in transform"
        )

    def expected_df_1(self, library="pandas"):
        """Expected output from test_expected_output_min_and_max."""
        df_dict = {
            "a": [2, 2, 3, 4, 5, 5, None],
            "b": [1, 2, 3, None, 7, 7, 7],
            "c": [None, 1, 2, 3, 0, 0, 0],
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


class TestBaseCappingTransformerInit(GenericCappingInitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"


class TestBaseCappingTransformerFit(GenericCappingFitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"


class TestBaseCappingTransformerTransform(GenericCappingTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseCappingTransformer"


class TestWeightedQuantile:
    """Tests for the BaseCappingTransformer.weighted_quantile method."""

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("values", "sample_weight", "quantiles", "expected_quantiles"),
        [
            (
                [1, 2, 3],
                [1, 1, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
            ),
            (
                [1, 2, 3],
                [0, 1, 0],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ),
            (
                [1, 2, 3],
                [1, 1, 0],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            ),
            (
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            ),
            ([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [0, 0.5, 1.0], [1.0, 2.0, 5.0]),
        ],
    )
    def test_expected_output(
        self,
        values,
        sample_weight,
        quantiles,
        expected_quantiles,
        library,
    ):
        """Test that weighted_quantile gives the expected outputs."""
        x = BaseCappingTransformer(capping_values={"a": [2, 10]})

        values_col = "values"
        weights_col = "weight"
        df_dict = {
            values_col: values,
            weights_col: sample_weight,
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        actual = x.weighted_quantile(
            df,
            quantiles,
            values_column=values_col,
            weights_column=weights_col,
        )

        # round to 1dp to avoid mismatches due to numerical precision
        actual_rounded_1_dp = list(np.round(actual, 1))

        assert actual_rounded_1_dp == expected_quantiles, (
            "unexpected weighted quantiles calculated"
        )
