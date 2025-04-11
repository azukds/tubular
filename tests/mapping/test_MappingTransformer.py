import narwhals as nw
import pytest

import tests.test_data as d
from tests.mapping.test_BaseMappingTransformer import (
    BaseMappingTransformerInitTests,
    BaseMappingTransformerTransformTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.mapping import MappingTransformer


def expected_df_1(library="pandas"):
    """Expected output for test_expected_output."""

    df_dict = {"a": ["a", "b", "c", "d", "e", "f"], "b": [1, 2, 3, 4, 5, 6]}

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)

    df = df.with_columns(nw.col("b").cast(nw.Int8))

    return df.to_native()


def expected_df_2(library="pandas"):
    """Expected output for test_non_specified_values_unchanged."""

    df_dict = {"a": [5, 6, 7, 4, 5, 6], "b": ["z", "y", "x", "d", "e", "f"]}

    df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    df = nw.from_native(df)

    df = df.with_columns(nw.col("a").cast(nw.Int8))

    return df.to_native()


class TestInit(BaseMappingTransformerInitTests):
    """Tests for MappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"


class TestTransform(BaseMappingTransformerTransformTests):
    """Tests for the transform method on MappingTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output(self, library):
        """Test that transform is giving the expected output."""

        df = d.create_df_1(library=library)
        expected = expected_df_1(library=library)

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        return_dtypes = {"a": "String", "b": "Int8"}

        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = x.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_non_specified_values_unchanged(self, library):
        """Test that values not specified in mappings are left unchanged in transform."""

        df = d.create_df_1(library=library)
        expected = expected_df_2(library=library)

        mapping = {"a": {1: 5, 2: 6, 3: 7}, "b": {"a": "z", "b": "y", "c": "x"}}

        return_dtypes = {"a": "Int8", "b": "String"}

        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = x.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("mapping", "return_dtypes"),
        [
            ({"a": {1: 1.1, 6: 6.6}}, {"a": "Float64"}),
            (
                {"a": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}},
                {"a": "String"},
            ),
            (
                {"a": {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}},
                {"a": "Boolean"},
            ),
            (
                {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}},
                {"b": "Int32"},
            ),
            (
                {"b": {"a": 1.1, "b": 2.2, "c": 3.3, "d": 4.4, "e": 5.5, "f": 6.6}},
                {"b": "Float32"},
            ),
        ],
    )
    def test_expected_dtype_conversions(
        self,
        mapping,
        return_dtypes,
        library,
    ):
        df = d.create_df_1(library=library)
        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)
        df = x.transform(df)

        column = list(mapping.keys())[0]
        actual_dtype = str(nw.from_native(df).get_column(column).dtype)
        assert (
            actual_dtype == return_dtypes[column]
        ), f"dtype converted unexpectedly, expected {return_dtypes[column]} but got {actual_dtype}"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_category_dtype_is_conserved(self, library):
        """This is a separate test due to the behaviour of category dtypes.

        See documentation of transform method
        """
        df = d.create_df_1(library=library)
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").cast(nw.Categorical)).to_native()

        mapping = {"b": {"a": "aaa", "b": "bbb"}}
        return_dtypes = {"b": "Categorical"}

        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)
        df = x.transform(df)

        assert (
            nw.from_native(df).get_column("b").dtype == nw.Categorical
        ), "Categorical dtype not preserved for column b"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("mapping", "mapped_col", "return_dtypes"),
        [
            ({"a": {99: 99, 98: 98}}, "a", {"a": "Int32"}),
            ({"b": {"z": "99", "y": "98"}}, "b", {"b": "String"}),
        ],
    )
    def test_no_applicable_mapping(self, mapping, mapped_col, return_dtypes, library):
        df = d.create_df_1(library=library)

        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        with pytest.warns(
            UserWarning,
            match=f"MappingTransformer: No values from mapping for {mapped_col} exist in dataframe.",
        ):
            x.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("mapping", "mapped_col", "return_dtypes"),
        [
            ({"a": {1: 1, 99: 99}}, "a", {"a": "Int64"}),
            ({"b": {"a": "1", "z": "99"}}, "b", {"b": "String"}),
        ],
    )
    def test_excess_mapping_values(self, mapping, mapped_col, return_dtypes, library):
        df = d.create_df_1(library=library)

        x = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        with pytest.warns(
            UserWarning,
            match=f"MappingTransformer: There are values in the mapping for {mapped_col} that are not present in the dataframe",
        ):
            x.transform(df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"
