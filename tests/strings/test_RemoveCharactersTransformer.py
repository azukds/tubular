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
from tubular.strings import RemoveCharactersTransformer


class TestRemoveCharactersTransformerInit(
    ColumnStrListInitTests,
):
    """Tests for RemoveCharactersTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RemoveCharactersTransformer"

    @pytest.mark.parametrize("bad_characters", [[1, 2], 1, True])
    def test_invalid_characters_error(
        self, minimal_attribute_dict, uninitialized_transformers, bad_characters
    ):
        """Test that an error is raised for invalid characters arg."""

        args = minimal_attribute_dict[self.transformer_name]
        args["characters"] = bad_characters
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestRemoveCharactersTransformerTransform(GenericTransformTests):
    "tests for RemoveCharactersTransformer.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RemoveCharactersTransformer"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("input_values", "expected_output", "characters"),
        [
            (["H     i"], ["Hi"], [r"\s"]),
            (
                ["Hello!!!", "<>Howdy?!?!", "[]hi()()"],
                ["Hello", "Howdy", "hi"],
                [r"\W"],
            ),
            (["Hello   ", "  Howdy?!", "hi!"], ["Hell", "Hwdy?!", "hi!"], [r"\s", "o"]),
        ],
    )
    def test_output_cases(
        self, lazy, library, from_json, input_values, expected_output, characters
    ):
        "test output cases for transformer"

        df_dict = {"a": input_values}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df_dict = {"a": expected_output}

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        transformer = RemoveCharactersTransformer(columns="a", characters=characters)

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_multi_column_output(self, lazy, library, from_json):
        "test multi column output case for transformer"

        df_dict = {"a": ["eighty 8", "99"], "b": ["hello123", "hi"]}

        expected_df_dict = {"a": ["eighty ", ""], "b": ["hello", "hi"]}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        transformer = RemoveCharactersTransformer(
            columns=["a", "b"], characters=[r"\d"]
        )

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for RemoveCharactersTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RemoveCharactersTransformer"
