import re

import narwhals as nw
import numpy as np
import pytest

import tests.test_data as d
from tests.base_tests import EmptyMappingsFitTransformPassTests, ReturnNativeTests
from tests.mapping.test_BaseMappingTransformer import (
    BaseMappingTransformerInitTests,
    BaseMappingTransformerTransformTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
    OtherBaseBehaviourTestsString,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
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


class TestTransform(BaseMappingTransformerTransformTests, ReturnNativeTests):
    """Tests for the transform method on MappingTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output(self, library, from_json, lazy):
        """Test that transform is giving the expected output."""

        df = d.create_df_1(library=library)
        expected = expected_df_1(library=library)

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        return_dtypes = {"a": "String", "b": "Int8"}

        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy=lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_null_type(self, library, from_json, lazy):
        """Test that transform is giving the expected output when type of mappings is null."""

        df_dict = {"a": [np.nan, None, 1]}
        expected_df_dict = {"a": [None, None, 1]}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        expected = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        mapping = {"a": {np.nan: None}}

        transformer = MappingTransformer(mappings=mapping)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy=lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_null_handling_for_str_return_type(self, library, from_json, lazy):
        """Test that transform is giving the expected output - for string type with nulls."""

        df_dict = {"a": [np.nan, None, 1]}
        expected_df_dict = {"a": [None, None, "b"]}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        expected = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        mapping = {"a": {1: "b"}}

        transformer = MappingTransformer(mappings=mapping)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy=lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_non_specified_values_unchanged(self, library, from_json, lazy):
        """Test that values not specified in mappings are left unchanged in transform."""

        df = d.create_df_1(library=library)
        expected = expected_df_2(library=library)

        mapping = {"a": {1: 5, 2: 6, 3: 7}, "b": {"a": "z", "b": "y", "c": "x"}}

        return_dtypes = {"a": "Int8", "b": "String"}

        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(df_transformed, lazy=lazy), expected)

        df = nw.from_native(df)
        expected = nw.from_native(expected)

        # also check single rows
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy=lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
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
                {
                    "b": {
                        "a": True,
                        "b": True,
                        "c": True,
                        "d": False,
                        "e": False,
                        "f": False,
                    }
                },
                {"b": "Boolean"},
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
        from_json,
        lazy,
    ):
        # skip test for pandas/boolean, which is blocked by validation
        if library == "pandas" and "Boolean" in return_dtypes.values():
            return

        df = d.create_df_1(library=library)
        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        output = _collect_frame(output, lazy=lazy)

        column = next(iter(mapping.keys()))
        actual_dtype = str(nw.from_native(output).get_column(column).dtype)
        assert actual_dtype == return_dtypes[column], (
            f"dtype converted unexpectedly, expected {return_dtypes[column]} but got {actual_dtype}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_category_dtype_is_conserved(self, library, from_json, lazy):
        """This is a separate test due to the behaviour of category dtypes.

        See documentation of transform method
        """
        df = d.create_df_1(library=library)
        df = nw.from_native(df)
        df = df.with_columns(nw.col("b").cast(nw.Categorical)).to_native()

        mapping = {"b": {"a": "aaa", "b": "bbb"}}
        return_dtypes = {"b": "Categorical"}

        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        output = _collect_frame(output, lazy=lazy)

        assert nw.from_native(output).get_column("b").dtype == nw.Categorical, (
            "Categorical dtype not preserved for column b"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("mapping", "return_dtype", "expected_values"),
        [
            ({"a": {"a": 1.0, "b": 2.0}}, "Float64", [1.0, 2.0]),
            ({"a": {"a": 1, "b": 2}}, "Int64", [1.0, 2.0]),
            ({"a": {"a": True, "b": False}}, "Boolean", [True, False]),
            ({"a": {"a": "x", "b": "y"}}, "String", ["x", "y"]),
            ({"a": {"a": "x", "b": "y"}}, "Categorical", ["x", "y"]),
        ],
    )
    def test_expected_output_for_categorical_input(
        self,
        library,
        from_json,
        lazy,
        mapping,
        return_dtype,
        expected_values,
    ):
        """Test that categorical columns can be mapped successfully."""

        # skip test for pandas/boolean, which is blocked by validation
        if library == "pandas" and return_dtype == "Boolean":
            return

        df_dict = {"a": ["a", "b"]}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        df = nw.from_native(df)
        df = df.with_columns(nw.col("a").cast(nw.Categorical)).to_native()

        expected_dict = {
            "a": expected_values,
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col("a").cast(getattr(nw, return_dtype))
        ).to_native()

        # convert bool type to pyarrow
        if library == "pandas" and return_dtype == "Boolean":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                nw.maybe_convert_dtypes(expected_df["a"])
            )
            expected_df = expected_df.to_native()

        return_dtypes = {"a": return_dtype}
        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(
            expected_df, _collect_frame(df_transformed, lazy=lazy)
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_boolean_with_nulls(self, library, from_json, lazy):
        """Test that output is as expected for tricky bool cases:
        e.g. mapping {True:1, False:0, None: 0}, potential causes of failure:
            - None being cast to False when these values are inserted into bool series
            - None mapping failing, as mapping logic relies on merging and None->None values
            will not merge

        Example failure 1:
        df=pd.DataFrame({'a': [True, False, None]})
        mappings={True:1, False:0, None:0}
        return_dtypes={'a': 'Int8'}
        mapping_transformer=MappingTransformer(mappings, return_dtypes)

        mapping_transformer.transform(df)->
        pd.DataFrame(
            {
            'a': [
                1,
                0,
                None # mapping merge has failed on None,
                #resulting in None instead of 0
            ]
            }
        )

        ---------
        Example Failure 2
        df=pd.DataFrame({'a': [1, 0, -1]})
        mappings={1:True, 0:False, -1:None}
        return_dtypes={'a': 'Int8'}
        mapping_transformer=MappingTransformer(mappings, return_dtypes)

        mapping_transformer.transform(df)->
        pd.DataFrame(
            {
            'a': [
                True,
                False,
                # when the mapping values are put into bool series
                # the none value is converted to False, instead of None
                False,

            ]
            }
        )

        """

        df_dict = {
            "a": [None, 0, 1, None, 0],
            "b": [True, False, None, True, False],
            "c": [None, 0, 0, None, 1],
            "d": [True, None, None, True, False],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        if library == "pandas":
            df = df.convert_dtypes()
            # df["b"]=df["b"].convert_dtypes()
            # df["d"]=df["d"].convert_dtypes()

        mapping = {
            "a": {0: False, 1: True},
            "b": {False: 0, True: 1},
            "c": {0: False, None: False, 1: True},
            "d": {False: 1, True: 0, None: 1},
        }

        return_dtypes = {
            "a": "Boolean",
            "b": "Float64",
            "c": "Boolean",
            "d": "Int64",
        }

        expected_dict = {
            "a": [None, False, True, None, False],
            "b": [1.0, 0.0, None, 1.0, 0.0],
            "c": [False, False, False, False, True],
            "d": [0, 1, 1, 0, 1],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        # convert bool type to pyarrow
        if library == "pandas":
            expected = expected.convert_dtypes()
            expected["b"] = expected["b"].astype("Float64")

        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(expected, _collect_frame(df_transformed, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("input_values", "input_type"),
        [
            (["01/02/2020", "20/03/1990"], "Datetime"),
            # test these just for pandas, should error for old non nullable bool types
            ([True, None], "Object"),
            ([True, False], "bool"),
        ],
    )
    def test_error_for_bad_type(
        self, library, from_json, lazy, input_values, input_type
    ):
        """Test expected error is raised for unexpected type."""

        if library == "polars" and input_type in {"bool", "Object"}:
            return

        df_dict = {"a": input_values}
        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        df = nw.from_native(df)
        if input_type == "Datetime":
            df = df.with_columns(nw.col("a").str.to_datetime(format="%d/%m/%Y"))
        schema = df.schema
        df = df.to_native()

        mapping = {"a": {"a": "aaa"}}
        return_dtypes = {"a": "Categorical"}

        transformer = MappingTransformer(mappings=mapping, return_dtypes=return_dtypes)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer = _handle_from_json(transformer, from_json)

        if input_type == "bool":
            bad_bool_type_cols = ["a"]
            msg = f"MappingTransformer: Older pandas boolean dtypes (bool, object) are no longer supported, please us pd.DataFrame.convert_dtypes to convert to nullable boolean type of columns {bad_bool_type_cols}."

        else:
            bad_types = {"a": schema["a"]}
            msg = rf"MappingTransformer: The following columns have types which are not covered by the existing mapping logic {bad_types}"

        with pytest.raises(
            TypeError,
            match=re.escape(msg),
        ):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyMappingsFitTransformPassTests,
    OtherBaseBehaviourTestsString,
):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"
