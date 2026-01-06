import narwhals as nw
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.misc import RenameColumnsTransformer


class TestInit(ColumnStrListInitTests):
    """Generic tests for RenameColumnsTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RenameColumnsTransformer"

    def test_invalid_new_column_names_error(self):
        "test expected error thrown if columns/new_column_names don't align"
        msg = "RenameColumnsTransformer: all provided columns must appear as keys in new_column_names"
        with pytest.raises(ValueError, match=msg):
            RenameColumnsTransformer(columns=["a"], new_column_names={"b": "a"})


class TestFit(GenericFitTests):
    """Generic tests for RenameColumnsTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RenameColumnsTransformer"


class TestTransform(GenericTransformTests):
    """Tests for RenameColumnsTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RenameColumnsTransformer"

    @pytest.mark.parametrize(
        "drop_original",
        [True, False],
    )
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize(
        "from_json",
        [True, False],
    )
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_output(
        self,
        library,
        from_json,
        lazy,
        drop_original,
    ):
        "test outputs for RenameColumnsTransformer.transform"

        transformer = RenameColumnsTransformer(
            columns=["a", "b"],
            new_column_names={"a": "new_a", "b": "new_b"},
            drop_original=drop_original,
        )

        df_dict = {"a": [1, None, 3], "b": ["x", "y", None]}

        expected_df_dict = {
            "new_a": df_dict["a"],
            "new_b": df_dict["b"],
        }

        if not drop_original:
            expected_df_dict.update(df_dict)

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        transformer = _handle_from_json(transformer, from_json=from_json)

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        # align column order
        expected = expected[output.columns]

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected)

        # also test single rows
        df = nw.from_native(df)
        expected = nw.from_native(expected)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy=lazy),
                df_expected_row,
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for RenameColumnsTransformer behaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RenameColumnsTransformer"
