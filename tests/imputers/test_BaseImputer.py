from copy import deepcopy

import narwhals as nw
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _check_if_skip_test,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)

# Categorical columns created under the same global string cache have the same underlying
# physical value when string values are equal.
# there is an efficiency cost, but not an issue for tests
pl.enable_string_cache()


class GenericImputerTransformTests:
    @pytest.fixture()
    def test_fit_df(self, request):
        library = request.param
        df_dict = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": ["a", "b", "c", "d", "e", "f", None],
            "c": ["a", "b", "c", "d", "e", "f", None],
        }

        return dataframe_init_dispatch(df_dict, library)

    @pytest.fixture()
    def expected_df_1(self, request):
        library = request.param
        df1_dict = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": ["a", "b", "c", "d", "e", "f", None],
            "c": ["a", "b", "c", "d", "e", "f", None],
        }

        df1 = dataframe_init_dispatch(df1_dict, library)

        narwhals_df = nw.from_native(df1)
        narwhals_df = narwhals_df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return narwhals_df.to_native()

    @pytest.fixture()
    def expected_df_2(self, request):
        library = request.param
        df2_dict = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None],
            "b": ["a", "b", "c", "d", "e", "f", "g"],
            "c": ["a", "b", "c", "d", "e", "f", None],
        }
        df2 = dataframe_init_dispatch(df2_dict, library)
        narwhals_df = nw.from_native(df2)
        narwhals_df = narwhals_df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return narwhals_df.to_native()

    @pytest.fixture()
    def expected_df_3(self, request):
        library = request.param
        df3_dict = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None],
            "b": ["a", "b", "c", "d", "e", "f", "g"],
            "c": ["a", "b", "c", "d", "e", "f", "f"],
        }

        df3 = dataframe_init_dispatch(dataframe_dict=df3_dict, library=library)

        narwhals_df = nw.from_native(df3)
        narwhals_df = narwhals_df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return narwhals_df.to_native()

    @pytest.fixture()
    def expected_df_4(self, request):
        library = request.param
        df4_dict = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None],
            "b": ["a", "b", "c", "d", "e", "f", "z"],
            "c": ["a", "b", "c", "d", "e", "f", "z"],
        }

        df4 = dataframe_init_dispatch(dataframe_dict=df4_dict, library=library)

        narwhals_df = nw.from_native(df4)
        narwhals_df = narwhals_df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return narwhals_df.to_native()

    @pytest.mark.parametrize("test_fit_df", ["pandas", "polars"], indirect=True)
    def test_not_fitted_error_raised(self, test_fit_df, initialized_transformers):
        if initialized_transformers[self.transformer_name].FITS:
            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(test_fit_df)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_impute_value_unchanged(self, library, initialized_transformers, from_json):
        """Test that self.impute_value is unchanged after transform."""
        df1 = d.create_df_1(library=library)
        transformer = initialized_transformers[self.transformer_name]

        impute_value = "g"
        transformer.impute_values_ = {"b": impute_value}

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = impute_value

        impute_values = deepcopy(transformer.impute_values_)

        if _check_if_skip_test(transformer, df1, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        transformer.transform(df1)

        assert transformer.impute_values_ == impute_values, (
            "impute_values_ changed in transform"
        )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_1"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_1"],
    )
    def test_expected_output_on_float_column(
        self,
        library,
        expected_df_1,
        initialized_transformers,
        from_json,
    ):
        """Test that transform is giving the expected output when applied to float column."""
        # Create the DataFrame using the library parameter
        df2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        transformer.impute_values_ = {"a": 7}

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = 7

        transformer.columns = ["a"]

        if _check_if_skip_test(transformer, df2, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        assert_frame_equal_dispatch(
            df_transformed,
            expected_df_1,
        )
        df2 = nw.from_native(df2)
        expected_df_1 = nw.from_native(expected_df_1)

        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_1[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_2"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_2"],
    )
    def test_expected_output_on_object_column(
        self,
        library,
        expected_df_2,
        initialized_transformers,
        from_json,
    ):
        """Test that transform is giving the expected output when applied to object column."""
        # Create the DataFrame using the library parameter
        df2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        impute_value = "g"
        transformer.impute_values_ = {"b": impute_value}

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = impute_value

        transformer.columns = ["b"]

        if _check_if_skip_test(transformer, df2, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        assert_frame_equal_dispatch(
            df_transformed,
            expected_df_2,
        )
        df2 = nw.from_native(df2)
        expected_df_2 = nw.from_native(expected_df_2)

        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_2[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_3", "impute_values_dict"),
        [
            ("pandas", "pandas", {"b": "g", "c": "f"}),
            ("polars", "polars", {"b": "g", "c": "f"}),
        ],
        indirect=["expected_df_3"],
    )
    def test_expected_output_with_object_and_categorical_columns(
        self,
        library,
        expected_df_3,
        initialized_transformers,
        impute_values_dict,
        from_json,
    ):
        """Test that transform is giving the expected output when applied to object and categorical columns."""
        # Create the DataFrame using the library parameter
        df2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        transformer.impute_values_ = impute_values_dict

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = "f"

        transformer.columns = ["b", "c"]

        if _check_if_skip_test(transformer, df2, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        assert_frame_equal_dispatch(
            df_transformed,
            expected_df_3,
        )
        df2 = nw.from_native(df2)
        expected_df_3 = nw.from_native(expected_df_3)

        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_3[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    @pytest.mark.parametrize(
        ("column", "impute_value", "expected"),
        [
            ("a", False, [True, False, False]),
            ("b", 0, [1, 2, 0]),
        ],
    )
    def test_imputation_with_falsey_values(
        self,
        library,
        initialized_transformers,
        column,
        impute_value,
        expected,
        from_json,
    ):
        """Test that transform is giving the expected output when imputation value is falsey."""
        # Create the DataFrame using the library parameter
        df_dict = {
            "a": [True, False, None],
            "b": [1, 2, None],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)

        # pandas bool+null cols default to Object, cast to bool
        df = df.with_columns(nw.col("a").cast(nw.Boolean))
        df = nw.to_native(df)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = impute_value

        transformer.columns = [column]

        transformer.impute_values_ = dict.fromkeys(transformer.columns, impute_value)

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        expected_df_dict = {
            column: expected,
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict,
            library=library,
        )

        # make sure types align with original
        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col(column).cast(nw.from_native(df)[column].dtype),
        )

        # Transform the DataFrame
        df_transformed = transformer.transform(df)

        assert_frame_equal_dispatch(
            df_transformed[[column]],
            expected_df.to_native()[[column]],
        )


class GenericImputerTransformTestsWeight:
    @pytest.fixture()
    def expected_df_weights(self, request):
        """Expected output for test_nulls_imputed_correctly_weights."""
        library = request.param
        df = d.create_df_9(library=library)

        df = nw.from_native(df)

        df = df.with_columns(df["b"].fill_null(4))

        return df.to_native()

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_weights"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_weights"],
    )
    def test_nulls_imputed_correctly_weights(
        self,
        library,
        expected_df_weights,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        # Create the DataFrame using the library parameter
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weights_column"] = "c"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Set the impute values dict directly rather than fitting x on df so test works with helpers
        impute_value = 4
        transformer.impute_values_ = {"b": impute_value}

        if self.transformer_name == "ArbitraryImputer":
            self.impute_value = impute_value

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(df)

        # Convert both DataFrames to a common format using Narwhals
        df_transformed_common = nw.from_native(df_transformed)
        expected_df_weights_common = nw.from_native(expected_df_weights)

        # Check outcomes for single rows
        for i in range(len(df_transformed_common)):
            df_transformed_row = df_transformed_common[[i]].to_native()
            df_expected_row = expected_df_weights_common[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

        # Check whole dataframes
        assert_frame_equal_dispatch(
            df_transformed_common.to_native(),
            expected_df_weights_common.to_native(),
        )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified_weights(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = ["a", "b"]
        args["weights_column"] = "c"

        transformer1 = uninitialized_transformers[self.transformer_name](**args)

        transformer1.fit(df)

        if _check_if_skip_test(transformer1, df, lazy=False, from_json=from_json):
            return

        transformer1 = _handle_from_json(transformer1, from_json)

        transformer2 = uninitialized_transformers[self.transformer_name](**args)

        transformer2.fit(df)

        transformer2 = _handle_from_json(transformer2, from_json)

        transformer2.transform(df)

        # Check if the impute_values_ are the same
        assert transformer1.impute_values_ == transformer2.impute_values_, (
            f"Impute values changed in transform for {self.transformer_name}"
        )


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestTransform(GenericImputerTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    # overload test as class needs special  handling to run
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=["minimal_dataframe_lookup"],
    )
    def test_get_feature_names_out_matches_new_features(
        self,
        minimal_dataframe_lookup,
        initialized_transformers,
    ):
        """Test that the expected newly created features (if any) are indeed contained
        in the output df"""

        df = minimal_dataframe_lookup[self.transformer_name]

        x = initialized_transformers[self.transformer_name]

        x.impute_values_ = dict.fromkeys(x.columns, 1)

        output = x.transform(df)

        output_columns = set(output.columns)

        expected_new_columns = set(x.get_feature_names_out())

        # are expected columns in the data
        assert expected_new_columns.intersection(output_columns), (
            f"{x.classname()}: get_feature_names_out does not agree with output of .transform, expected {expected_new_columns} but got {output_columns}"
        )

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"
