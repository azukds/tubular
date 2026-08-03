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
from tubular.imputers import CategoricalImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CategoricalImputer"

    def test_bad_impute_value_error(self):
        pass


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CategoricalImputer"


class TestTransform(
    GenericTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CategoricalImputer"

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

        transformer = CategoricalImputer(impute_value="a", columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        bad_types = [nw.from_native(df).schema[column]]

        msg = f"""
                ArbitraryImputer: transformer can only handle Categorical/Unknown type columns
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
    @pytest.mark.parametrize(
        ("input_col", "impute_value", "expected_values"),
        [
            (["a", None, "c"], "b", ["a", "b", "c"]),
            ([None, None, None], "missing", ["missing", "missing", "missing"]),
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

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df_nw = nw.from_native(df)

        dtype = nw.Categorical

        df_nw = df_nw.with_columns(nw.col("a").cast(dtype))

        transformer = CategoricalImputer(impute_value=impute_value, columns=[column])

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
        expected_dtype = "Categorical"
        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected_dtype = nw.Categorical
        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library).cast(
                expected_dtype,
            ),
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

        impute_value = "c"
        expected_type = "Categorical"
        transformer = CategoricalImputer(impute_value=impute_value, columns=[column])

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
        cls.transformer_name = "CategoricalImputer"
