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
from tubular.strings import ExtractStringComponentsTransformer


class TestExtractStringComponentsTransformerInit(
    ColumnStrListInitTests,
):
    """Tests for ExtractStringComponentsTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ExtractStringComponentsTransformer"

    @pytest.mark.parametrize("bad_by", [[1, 2], 1, True])
    def test_invalid_by_error(
        self, minimal_attribute_dict, uninitialized_transformers, bad_by
    ):
        """Test that an error is raised for invalid by arg."""

        args = minimal_attribute_dict[self.transformer_name]
        args["by"] = bad_by
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("bad_return_n_components", [[1, 2], "a", {}])
    def test_invalid_return_n_components_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        bad_return_n_components,
    ):
        """Test that an error is raised for invalid return_n_components arg."""

        args = minimal_attribute_dict[self.transformer_name]
        args["return_n_components"] = bad_return_n_components
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestExtractStringComponentsTransformerTransform(GenericTransformTests):
    "tests for ExtractStringComponentsTransformer.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ExtractStringComponentsTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("input_values", "by", "return_n_components", "expected_output"),
        [
            (["hi.hello"], ".", 2, [["hi"], ["hello"]]),
            ([None], "p", 3, [[None], [None], [None]]),
            (
                ["hi how", "are you ?", None],
                " ",
                3,
                [["hi", "are", None], ["how", "you", None], [None, "?", None]],
            ),
            (
                ["greg@apple.com", "tom@mac.net"],
                "@",
                2,
                [["greg", "tom"], ["apple.com", "mac.net"]],
            ),
        ],
    )
    def test_output_cases(
        self,
        lazy,
        library,
        from_json,
        input_values,
        by,
        return_n_components,
        expected_output,
    ):
        "test output cases for transformer"

        df_dict = {"a": input_values}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df_dict = {
            "a": input_values,
            **{
                f"a_split_by_{by}_entry_{i}": expected_output[:][i]
                for i in range(return_n_components)
            },
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        if library == "pandas":
            df = df.convert_dtypes(dtype_backend="pyarrow")
            expected_df = expected_df.convert_dtypes(dtype_backend="pyarrow")

        # cast for the single row null case
        df = nw.from_native(df)
        df = df.with_columns(nw.col("a").cast(nw.String)).to_native()

        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col(col).cast(nw.String) for col in expected_df.columns
        ).to_native()

        transformer = ExtractStringComponentsTransformer(
            columns="a", by=by, return_n_components=return_n_components
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_output_multi_column_case(self, lazy, library, from_json):
        "test output cases for transformer"

        df_dict = {"a": ["hi.bye", "bye.hi.car"], "b": ["cat.dog", "dog.mouse"]}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df_dict = {
            **df_dict,
            "a_split_by_._entry_0": ["hi", "bye"],
            "a_split_by_._entry_1": ["bye", "hi"],
            "a_split_by_._entry_2": [None, "car"],
            "b_split_by_._entry_0": ["cat", "dog"],
            "b_split_by_._entry_1": ["dog", "mouse"],
            "b_split_by_._entry_2": [None, None],
        }

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        if library == "pandas":
            df = df.convert_dtypes(dtype_backend="pyarrow")
            expected_df = expected_df.convert_dtypes(dtype_backend="pyarrow")

        expected_df = nw.from_native(expected_df)
        expected_df = expected_df.with_columns(
            nw.col(col).cast(nw.String) for col in expected_df.columns
        ).to_native()

        transformer = ExtractStringComponentsTransformer(
            columns=["a", "b"], by=".", return_n_components=3
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for ExtractStringComponentsTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ExtractStringComponentsTransformer"
