import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
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
from tubular.misc import ColumnDtypeSetter


class TestInit(ColumnStrListInitTests):
    """Generic tests for ColumnDtypeSetter.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"

    @pytest.mark.parametrize(
        "invalid_dtype",
        ["STRING", "misc_invalid", "np.int", 0],
    )
    def test_invalid_dtype_error(self, invalid_dtype):
        with pytest.raises(BeartypeCallHintParamViolation):
            ColumnDtypeSetter(columns=["a"], dtype=invalid_dtype)


class TestFit(GenericFitTests):
    """Generic tests for ColumnDtypeSetter.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"


class TestTransform(GenericTransformTests):
    """Tests for ColumnDtypeSetter.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"

    @pytest.mark.parametrize(
        ("a_values", "b_values", "dtype", "expected_a_values", "expected_b_values"),
        [
            ([1, 2], [3, 4], "Float64", [1.0, 2.0], [3.0, 4.0]),
            ([1, 2], [3, 4], "Int16", [1.0, 2.0], [3.0, 4.0]),
            ([1, 2], [3, 4], "UInt16", [1.0, 2.0], [3.0, 4.0]),
            ([0, 1], [None, 1], "Boolean", [False, True], [None, True]),
            ([True, False], [None, True], "Float32", [1.0, 0.0], [None, 1.0]),
            ([True, False], [True, True], "Int32", [1, 0], [1, 1]),
            ([True, False], [True, True], "UInt64", [1, 0], [1, 1]),
            (["a", "b"], ["c", "d"], "String", ["a", "b"], ["c", "d"]),
            (["a", "b"], ["c", "d"], "Categorical", ["a", "b"], ["c", "d"]),
            ([None, None], [None, None], "Float32", [None, None], [None, None]),
            ([None, None], [None, None], "String", [None, None], [None, None]),
            ([None, None], [None, None], "Categorical", [None, None], [None, None]),
            ([None, None], [None, None], "Boolean", [None, None], [None, None]),
        ],
    )
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    def test_expected_output(
        self,
        a_values,
        b_values,
        expected_a_values,
        expected_b_values,
        dtype,
        lazy,
        from_json,
        library,
    ):
        """Test values are cast to correct dtype."""

        transformer = ColumnDtypeSetter(columns=["a", "b"], dtype=dtype)

        df_dict = {"a": a_values, "b": b_values}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        expected_df_dict = {"a": expected_a_values, "b": expected_b_values}

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )
        expected_df = nw.from_native(expected_df)

        if library == "pandas" and transformer.dtype == "Boolean":
            expected_df = expected_df.with_columns(
                nw.maybe_convert_dtypes(expected_df[col]).cast(nw.Boolean)
                for col in transformer.columns
            ).to_native()
        else:
            expected_df = expected_df.with_columns(
                nw.col(col).cast(getattr(nw, transformer.dtype))
                for col in transformer.columns
            ).to_native()

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json=from_json)

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(
            _collect_frame(df_transformed, lazy=lazy), expected_df
        )

        # Test single row transformation
        df = nw.from_native(df)
        expected_df = nw.from_native(expected_df)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy)
            )
            df_expected_row = expected_df[[i]]
            # special handling for category type to account for single row/one category
            if transformer.dtype == "Categorical":
                df_expected_row = df_expected_row.with_columns(
                    nw.new_series(
                        name=col,
                        values=[expected_df_dict[col][i]],
                        backend=library,
                    ).cast(nw.Categorical)
                    for col in transformer.columns
                )

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row.to_native(),
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for ColumnDtypeSetter behaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ColumnDtypeSetter"
