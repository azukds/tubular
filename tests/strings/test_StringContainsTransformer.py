import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.strings import StringContainsTransformer


class TestStringContainsTransformerInit(
    ColumnStrListInitTests,
):
    """Tests for StringContainsTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringContainsTransformer"

    @pytest.mark.parametrize("bad_reference", [[1, 2], 1, True])
    def test_invalid_reference_error(
        self, minimal_attribute_dict, uninitialized_transformers, bad_reference
    ):
        """Test that an error is raised for invalid characters arg."""

        args = minimal_attribute_dict[self.transformer_name]
        args["reference"] = bad_reference
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("bad_reference_as_column", [[1, 2], 1, "a"])
    def test_invalid_reference_as_column_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        bad_reference_as_column,
    ):
        """Test that an error is raised for invalid characters arg."""

        args = minimal_attribute_dict[self.transformer_name]
        args["reference_as_column"] = bad_reference_as_column
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestStringContainsTransformerTransform(GenericTransformTests):
    "tests for StringContainsTransformer.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringContainsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("input_values", "reference_values", "expected_output"),
        [
            (["a"], ["a"], [True]),
            ([None], [None], [None]),
            (["a"], ["b"], [False]),
            (["a", "b", None], ["b", "b", None], [False, True, None]),
        ],
    )
    def test_output_cases_as_column(
        self, lazy, library, from_json, input_values, reference_values, expected_output
    ):
        "test output cases for transformer when reference_column is True"

        df_dict = {"a": input_values, "b": reference_values}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        df = nw.from_native(df)
        df = df.with_columns(nw.col("a").cast(nw.String)).to_native()

        expected_df_dict = {
            "a": input_values,
            "b": reference_values,
            "a_contains_b": expected_output,
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )
        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col("a").cast(nw.String), nw.col("a_contains_b").cast(nw.Boolean)
        ).to_native()

        transformer = StringContainsTransformer(
            columns="a", reference="b", reference_as_column=True
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_output_cases_as_column_errors_for_pandas(self, lazy, from_json):
        "test expected error is thrown when reference_as_column is True and df is pandas"

        df_dict = {"a": ["bla", "blaa"], "b": ["hi", "hello"]}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library="pandas")

        transformer = StringContainsTransformer(
            columns="a", reference="b", reference_as_column=True
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        msg = "StringContainsTransformer: reference_as_column=True is only supported for polars backend"
        with pytest.raises(TypeError, match=msg):
            transformer.transform(_convert_to_lazy(df, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("input_values", "reference_value", "expected_output"),
        [
            (["a"], "a", [True]),
            ([None], "a", [None]),
            (["a"], "b", [False]),
            (["a", "b", None], "b", [False, True, None]),
        ],
    )
    def test_output_cases_as_value(
        self, lazy, library, from_json, input_values, reference_value, expected_output
    ):
        "test output cases for transformer when reference_column is True"

        df_dict = {"a": input_values}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)
        df = nw.from_native(df)
        df = df.with_columns(nw.col("a").cast(nw.String)).to_native()

        expected_df_dict = {
            "a": input_values,
            f"a_contains_{reference_value}": expected_output,
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )
        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col("a").cast(nw.String),
            nw.col(f"a_contains_{reference_value}").cast(nw.Boolean),
        ).to_native()

        transformer = StringContainsTransformer(
            columns="a", reference=reference_value, reference_as_column=False
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_multi_column_output_as_column(self, lazy, library, from_json):
        "test multi column output case for transformer when reference_as_column True"

        df_dict = {"a": ["1", "2"], "b": ["3", "4"], "c": ["1", "4"]}

        expected_df_dict = {
            **df_dict,
            "a_contains_c": [True, False],
            "b_contains_c": [False, True],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        transformer = StringContainsTransformer(
            columns=["a", "b"],
            reference="c",
            reference_as_column=True,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_multi_column_output_as_value(self, lazy, library, from_json):
        "test multi column output case for transformer when reference_as_column False"

        df_dict = {"a": ["cat", "dog"], "b": ["dog", "cat"]}

        expected_df_dict = {
            **df_dict,
            "a_contains_cat": [True, False],
            "b_contains_cat": [False, True],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        transformer = StringContainsTransformer(
            columns=["a", "b"],
            reference="cat",
            reference_as_column=False,
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for StringContainsTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringContainsTransformer"
