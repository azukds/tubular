import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    EmptyColumnsFitTransformPassTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
)
from tubular.imputers import BooleanImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BooleanImputer"

    def test_bad_impute_value_error(self):
        pass


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BooleanImputer"


class TestTransform(
    GenericTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BooleanImputer"

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "input_values",
        [
            ["a", "b"],
            [1.0, 2.0],
        ],
    )
    def test_type_mismatch_errors(
        self,
        input_values,
        library,
        lazy,
        from_json,
    ):
        """Test that unexpected dtypes will hit error"""

        column = "a"
        df_dict = {column: input_values}

        # because of weird types, initialise manually
        df = pd.DataFrame(df_dict) if library == "pandas" else pl.DataFrame(df_dict)

        transformer = BooleanImputer(impute_value=False, columns=[column])

        if _check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        bad_types = [nw.from_native(df).schema[column]]

        allowed_types_str = "Boolean/Unknown"
        if library == "pandas":
            allowed_types_str += "/Object"
        msg = f"""
                ArbitraryImputer: transformer can only handle {allowed_types_str} type columns
                but got columns with types {bad_types}
                """

        with pytest.raises(
            TypeError,
            match=re.escape(msg),
        ):
            transformer.transform(_convert_to_lazy(df, lazy))

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("input_col", "impute_value", "expected_values"),
        [
            ([True, False, None], True, [True, False, True]),
            ([True, False, None], False, [True, False, False]),
            ([None, None, None], True, [True, True, True]),
            ([None, None, None], False, [False, False, False]),
        ],
    )
    def test_output(
        self,
        input_col,
        impute_value,
        expected_values,
        library,
        lazy,
        from_json,
    ):
        """Test outputs for expected cases."""

        column = "a"

        # tricky to get all null column to correct type, as casting to boolean
        # maps None->False for pandas
        if library == "polars":
            df = pl.DataFrame({"a": input_col}, schema={"a": pl.Boolean})

        else:
            df = pd.DataFrame({"a": input_col}, dtype="boolean")

        df_nw = nw.from_native(df)

        transformer = BooleanImputer(impute_value=impute_value, columns=[column])

        if _check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(
            _convert_to_lazy(df_nw.to_native(), lazy),
        )

        df_transformed_nw = nw.from_native(
            _collect_frame(df_transformed_native, lazy),
        )

        actual_dtype = str(df_transformed_nw[column].dtype)
        expected_dtype = "Boolean"
        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()

        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library)
        )

        if library == "pandas":
            expected = expected.with_columns(nw.maybe_convert_dtypes(expected["a"]))

        assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
        )

        # Check outcomes for single rows
        expected = nw.from_native(expected)
        for i in range(len(df_nw)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df_nw[[i]].to_native(), lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("input_col", "impute_value", "expected_values"),
        [
            ([True, False, None], True, [True, False, True]),
            ([True, False, None], False, [True, False, False]),
        ],
    )
    def test_output_for_pandas_object_type(
        self,
        input_col,
        impute_value,
        expected_values,
        from_json,
    ):
        """Test outputs for pandas object type columns."""

        column = "a"

        # tricky to get all null column to correct type, as casting to boolean
        # maps None->False for pandas
        df = pd.DataFrame({"a": input_col})
        df = df.convert_dtypes()

        transformer = BooleanImputer(impute_value=impute_value, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        df_transformed = transformer.transform(df)

        expected = pd.DataFrame({"a": expected_values})
        expected = expected.convert_dtypes()

        assert_frame_equal_dispatch(
            expected,
            df_transformed,
        )

        # Check outcomes for single rows
        for i in range(len(df)):
            df_row = df.iloc[[i]]
            df_transformed_row = transformer.transform(df_row)
            df_expected_row = expected.iloc[[i]]

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    def test_polars_unknown_type_output(
        self,
        lazy,
        from_json,
    ):
        """Test handling of polars Unknown type column (output type should be inferred from impute_value)

        Test separately to pandas, as pandas does not have equivalent type.
        """

        column = "a"
        values = [None, None]
        df_dict = {"a": values}

        df = pl.DataFrame(df_dict)

        df_nw = nw.from_native(df)

        impute_value = True
        expected_dtype = "Boolean"
        transformer = BooleanImputer(impute_value=impute_value, columns=[column])

        if _check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(
            _convert_to_lazy(df_nw.to_native(), lazy),
        )

        df_transformed_nw = nw.from_native(
            _collect_frame(df_transformed_native, lazy),
        )

        actual_dtype = str(df_transformed_nw[column].dtype)

        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected = expected.with_columns(
            nw.new_series(
                name=column,
                values=[impute_value, impute_value],
                backend="polars",
            ).cast(getattr(nw, expected_dtype)),
        )

        assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
        )

        # Check outcomes for single rows
        expected = nw.from_native(expected)
        for i in range(len(df_nw)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df_nw[[i]].to_native(), lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row,
            )


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyColumnsFitTransformPassTests,
):
    """
    Class to run tests for Transformer behaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BooleanImputer"
