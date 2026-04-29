from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)


class TestExtractStringComponentsTransformerInit(
    ColumnStrListInitTests,
):
    """Tests for ExtractStringComponentsTransformer initialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ExtractStringComponentsTransformer"


class TestExtractStringComponentsTransformerTransform(GenericTransformTests):
    "tests for ExtractStringComponentsTransformer.transform"

    # @classmethod
    # def setup_class(cls):
    #     cls.transformer_name = "ExtractStringComponentsTransformer"

    # @pytest.mark.parametrize("lazy", [True, False])
    # @pytest.mark.parametrize("library", ["pandas", "polars"])
    # @pytest.mark.parametrize("from_json", [True, False])
    # @pytest.mark.parametrize(
    #     ("input_values", "expected_output"),
    #     [
    #         (["AaAaa"], ["aaaaa"]),
    #         (["HeLLO!!!  Hi", "  Howdy", "hi"], ["hello!!!  hi", "  howdy", "hi"]),
    #     ],
    # )
    # def test_output_cases(
    #     self, lazy, library, from_json, input_values, expected_output
    # ):
    #     "test output cases for transformer"

    #     df_dict = {"a": input_values}

    #     df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    #     expected_df_dict = {"a": expected_output}

    #     expected_df = dataframe_init_dispatch(
    #         dataframe_dict=expected_df_dict, library=library
    #     )

    #     transformer = ExtractStringComponentsTransformer(columns="a")

    #     if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
    #         return

    #     transformer = _handle_from_json(transformer, from_json=from_json)

    #     output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

    #     assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)

    # @pytest.mark.parametrize("lazy", [True, False])
    # @pytest.mark.parametrize("library", ["pandas", "polars"])
    # @pytest.mark.parametrize("from_json", [True, False])
    # def test_multi_column_output(self, lazy, library, from_json):
    #     "test multi column output case for transformer"

    #     df_dict = {
    #         "a": ["StAR WARs", "StaR Trek"],
    #         "b": ["GaME of Thrones", "Lord of the RINGS"],
    #     }

    #     expected_df_dict = {
    #         "a": ["star wars", "star trek"],
    #         "b": ["game of thrones", "lord of the rings"],
    #     }

    #     df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    #     expected_df = dataframe_init_dispatch(
    #         dataframe_dict=expected_df_dict, library=library
    #     )

    #     transformer = ExtractStringComponentsTransformer(columns=["a", "b"])

    #     if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
    #         return

    #     transformer = _handle_from_json(transformer, from_json=from_json)

    #     output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

    #     assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for ExtractStringComponentsTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ExtractStringComponentsTransformer"
