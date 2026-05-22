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
    OtherBaseBehaviourTestsNumeric,
    ReturnNativeTests,
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tests.utils import _handle_from_json, dataframe_init_dispatch
from tubular.imputers import BooleanImputer


def impute_df_with_several_types(library="pandas"):
    """
    Fixture that returns a DataFrame with columns suitable for downcasting
    for both pandas and polars.
    """
    data = {
        "a": ["a", "b", "c", "d", None],
        "b": [1.0, 2.0, 3.0, 4.0, None],
        "c": [True, False, False, None, True],
    }

    return u.dataframe_init_dispatch(data, library)


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


def create_expected_df_3(library="pandas"):
    "expected df for transform test."
    expected_df_dict = {
        "a": [1, 2, 3, 4, 5, 6, None],
        "b": ["a", "b", "c", "d", "e", "f", None],
        "c": ["a", "b", "c", "d", "e", "f", "g"],
    }

    expected_df = dataframe_init_dispatch(
        dataframe_dict=expected_df_dict, library=library
    )

    narwhals_df = nw.from_native(expected_df)
    narwhals_df = narwhals_df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

    return narwhals_df.to_native()


class TestTransform(
    GenericImputerTransformTests,
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
        ("column", "col_type"),
        [
            ("a", "String"),
            ("a", "Categorical"),
            ("b", "Float32"),
        ],
    )
    def test_type_mismatch_errors(
        self,
        column,
        col_type,
        library,
        lazy,
        from_json,
    ):
        """Test that dtypes are preserved after imputation."""

        df = impute_df_with_several_types(library=library)

        df = nw.from_native(df)

        df = df.with_columns(
            nw.col(column).cast(getattr(nw, col_type)),
        )

        df = nw.to_native(df)

        transformer = BooleanImputer(impute_value=True, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        allowed_types_str = "Boolean/Unknown"
        col_dtype = getattr(nw, col_type)
        if library == "pandas":
            allowed_types_str += "/Object"
        msg = f"""
            ArbitraryImputer: transformer can only handle {allowed_types_str} type columns
            but got columns with types {[col_dtype]}
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
    def test_impute_value_preserve_dtype(
        self,
        library,
        lazy,
        from_json,
    ):
        """Test that dtypes are preserved after imputation."""

        column = "c"

        df = impute_df_with_several_types(library=library)

        df_nw = nw.from_native(df)

        transformer = BooleanImputer(impute_value=False, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(
            u._convert_to_lazy(df_nw.to_native(), lazy),
        )

        df_transformed_nw = nw.from_native(
            u._collect_frame(df_transformed_native, lazy),
        )

        expected_dtype = df_nw[column].dtype

        actual_dtype = df_transformed_nw[column].dtype

        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()

        expected_values = [True, False, False, False, True]

        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library).cast(
                nw.Boolean,
            ),
        )

        u.assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
        )

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("input_col", "expected_dtype", "impute_value", "expected_values"),
        [
            ([True, False, None], "Boolean", True, [True, False, True]),
        ],
    )
    def test_edge_cases(
        self,
        input_col,
        expected_dtype,
        impute_value,
        expected_values,
        library,
        lazy,
        from_json,
    ):
        """Test handling for some edge cases:
        - pandas object type
        """

        column = "a"
        df_dict = {"a": input_col}

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df_nw = nw.from_native(df)

        transformer = BooleanImputer(impute_value=impute_value, columns=[column])

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

        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library).cast(
                getattr(nw, expected_dtype),
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
    @pytest.mark.parametrize(
        ("impute_value", "impute_val_type"),
        [
            (True, "Boolean"),
        ],
    )
    def test_polars_unknown_type_output(
        self,
        impute_value,
        impute_val_type,
        lazy,
        from_json,
    ):
        """Test handling of polars Unknown type column (output type should be inferred from impute_value)"""

        column = "a"
        values = [None, None]
        df_dict = {"a": values}

        df = pl.DataFrame(df_dict)

        df_nw = nw.from_native(df)

        transformer = BooleanImputer(impute_value=impute_value, columns=[column])

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

        assert actual_dtype == impute_val_type, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {impute_val_type} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()
        expected = expected.with_columns(
            nw.new_series(
                name=column,
                values=[impute_value, impute_value],
                backend="polars",
            ).cast(getattr(nw, impute_val_type)),
        )

        u.assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_nw.to_native(),
        )

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
        ],
    )
    def test_weird_dtype_errors(
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

        transformer = BooleanImputer(impute_value=1, columns=[column])

        if u._check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        bad_types = [nw.from_native(df).schema[column]]

        msg = re.escape(
            f"""
                ArbitraryImputer: transformer can only handle Float/Int/UInt/Unknown type columns
                but got columns with types {bad_types}
                """,
        )

        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(u._convert_to_lazy(df, lazy))


class TestOtherBaseBehaviour(
    OtherBaseBehaviourTests,
    EmptyColumnsFitTransformPassTests,
    OtherBaseBehaviourTestsNumeric,
):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BooleanImputer"
