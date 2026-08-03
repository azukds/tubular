import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import tests.utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    EmptyColumnsFitTransformPassTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.utils import _handle_from_json
from tubular.imputers import NumberImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NumberImputer"

    def test_bad_impute_value_error(self):
        pass


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NumberImputer"


class TestTransform(
    GenericTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NumberImputer"

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "input_values",
        [
            [["a", "b"], ["c", "d"]],
            [{"a": 1}, {"b": 4}],
            ["a", "b"],
            [True, False],
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

        transformer = NumberImputer(impute_value=1, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        bad_types = [nw.from_native(df).schema[column]]

        msg = f"""
                ArbitraryImputer: transformer can only handle Float/Int/UInt/Unknown type columns
                but got columns with types {bad_types}
                """

        with pytest.raises(
            TypeError,
            match=re.escape(msg),
        ):
            transformer.transform(u._convert_to_lazy(df, lazy))

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_impute_value_not_changed_by_transform(
        self,
        library,
        lazy,
        from_json,
    ):
        """Test outputs for expected cases."""

        column = "a"
        input_col = [1.0, None]
        impute_value = 2
        df_dict = {"a": input_col}

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df_nw = nw.from_native(df)

        transformer = NumberImputer(impute_value=impute_value, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        _ = transformer.transform(
            u._convert_to_lazy(df_nw.to_native(), lazy),
        )

        # check impute value not changed in transform
        assert transformer.impute_value == impute_value, (
            "impute_values_ changed in transform"
        )

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("input_col", "impute_value", "expected_values"),
        [
            ([1.0, None, 3.0], 2, [1.0, 2.0, 3.0]),
            ([None, None, None], 2, [2.0, 2.0, 2.0]),
            ([1, None, 3], 2, [1.0, 2.0, 3.0]),
            # test with decimal numbers
            ([1.3, 2.2, None], 1.1, [1.3, 2.2, 1.1]),
            # test imputing with falsey value
            ([1, None, 3], 0, [1.0, 0.0, 3.0]),
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
        df_dict = {"a": input_col}
        expected_dtype = "Float64"

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df_nw = nw.from_native(df)

        if all(val is None for val in input_col):
            df_nw = df_nw.with_columns(nw.col("a").cast(getattr(nw, expected_dtype)))

        transformer = NumberImputer(impute_value=impute_value, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(
            u._convert_to_lazy(df_nw.to_native(), lazy),
        )

        # check impute value not changed in transform
        assert transformer.impute_value == impute_value, (
            "impute_values_ changed in transform"
        )

        df_transformed_nw = nw.from_native(
            u._collect_frame(df_transformed_native, lazy),
        )

        actual_dtype = str(df_transformed_nw[column].dtype)
        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library)
        )

        u.assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
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

        impute_value = 1
        expected_type = "Int32"
        transformer = NumberImputer(impute_value=impute_value, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(
            u._convert_to_lazy(df_nw.to_native(), lazy),
        )

        df_transformed_nw = nw.from_native(
            u._collect_frame(df_transformed_native, lazy),
        )

        actual_dtype = str(df_transformed_nw[column].dtype)

        assert actual_dtype == expected_type, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_type} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected = expected.with_columns(
            nw.new_series(
                name=column,
                values=[impute_value, impute_value],
                backend="polars",
            ).cast(getattr(nw, expected_type)),
        )

        u.assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
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
        cls.transformer_name = "NumberImputer"
