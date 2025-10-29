import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import tests.test_data as d
import tests.utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tests.utils import _handle_from_json
from tubular.imputers import ArbitraryImputer


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
        cls.transformer_name = "ArbitraryImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"


class TestTransform(
    GenericImputerTransformTests,
    GenericTransformTests,
    ReturnNativeTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("column", "col_type", "impute_value"),
        [
            ("a", "String", 1),
            ("a", "Categorical", True),
            ("b", "Float32", "bla"),
            ("c", "Boolean", 500),
        ],
    )
    def test_type_mismatch_errors(
        self,
        column,
        col_type,
        impute_value,
        library,
        from_json,
    ):
        """Test that dtypes are preserved after imputation."""

        df = impute_df_with_several_types(library=library)

        df = nw.from_native(df)

        df = df.with_columns(
            nw.col(column).cast(getattr(nw, col_type)),
        )

        df = nw.to_native(df)

        transformer = ArbitraryImputer(impute_value=impute_value, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        if col_type in ["Categorical", "String"]:
            msg_required_impute_value_type = "str"
            msg_col_type = "Categorical or String"

        elif col_type == "Boolean":
            msg_required_impute_value_type = "bool"
            msg_col_type = "Boolean"

        else:
            msg_required_impute_value_type = "numeric"
            msg_col_type = "Numeric"

        msg = rf"""
                {self.transformer_name}: Attempting to impute non-{msg_required_impute_value_type} value {transformer.impute_value} into
                {msg_col_type} type columns, this is not type safe,
                please use {msg_required_impute_value_type} impute_value for these columns
                \(this may require separate ArbitraryImputer instances for different column types\)
                """

        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("column", "col_type", "impute_value", "expected_values"),
        [
            ("a", "String", "z", ["a", "b", "c", "d", "z"]),
            ("a", "Categorical", "z", ["a", "b", "c", "d", "z"]),
            ("b", "Float32", 1, [1.0, 2.0, 3.0, 4.0, 1.0]),
        ],
    )
    def test_impute_value_preserve_dtype(
        self,
        column,
        col_type,
        impute_value,
        expected_values,
        library,
        from_json,
    ):
        """Test that dtypes are preserved after imputation."""

        df = impute_df_with_several_types(library=library)

        df_nw = nw.from_native(df)

        # or just downcast easier types (String comes in correct type so leave)
        if col_type in ["Float32", "Categorical"]:
            df_nw = df_nw.with_columns(
                nw.col(column).cast(getattr(nw, col_type)),
            )

        transformer = ArbitraryImputer(impute_value=impute_value, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(df_nw.to_native())

        df_transformed_nw = nw.from_native(df_transformed_native)

        expected_dtype = df_nw[column].dtype

        native_namespace = nw.get_native_namespace(df_nw).__name__

        # for pandas categorical are converted to enum
        if col_type == "Categorical" and native_namespace == "pandas":
            expected_dtype = nw.Enum

        actual_dtype = df_transformed_nw[column].dtype

        assert actual_dtype == expected_dtype, (
            f"{self.transformer_name}: dtype changed unexpectedly in transform, expected {expected_dtype} but got {actual_dtype}"
        )

        # also check full df against expectation
        expected = df_nw.clone()

        expected = expected.with_columns(
            nw.new_series(name=column, values=expected_values, backend=library).cast(
                getattr(nw, col_type),
            ),
        )

        u.assert_frame_equal_dispatch(
            expected.to_native(),
            df_transformed_native,
            # this turns off checks for category metadata like ordering
            # this transformer will convert an unordered pd categorical to ordered
            # so this is needed
            check_categorical=False,
        )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("input_col", "expected_dtype", "impute_value", "expected_values"),
        [
            ([None, None], "String", "a", ["a", "a"]),
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
        from_json,
    ):
        """Test handling for some edge cases:
        - pandas object type
        - all null column
        """

        column = "a"
        df_dict = {"a": input_col}

        df = u.dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df_nw = nw.from_native(df)

        transformer = ArbitraryImputer(impute_value=impute_value, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        # for pandas, the all null column is inferred as string type
        # for polars, it is Unknown type, which triggers a warning
        if library == "polars" and input_col == [None, None]:
            with pytest.warns(
                UserWarning,
                match=f"{self.transformer_name}: X contains all null columns { {column}!s}, types for these columns will be inferred as {type(transformer.impute_value)}",
            ):
                df_transformed_native = transformer.transform(df_nw.to_native())

        else:
            df_transformed_native = transformer.transform(df_nw.to_native())

        df_transformed_nw = nw.from_native(df_transformed_native)

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

        u.assert_frame_equal_dispatch(expected.to_native(), df_transformed_native)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("impute_value", "impute_val_type"),
        [
            (1, "Int32"),
            ("a", "String"),
            (True, "Boolean"),
        ],
    )
    def test_polars_unknown_type_output(self, impute_value, impute_val_type, from_json):
        """Test handling of polars Unknown type column (output type should be inferred from impute_value)"""

        column = "a"
        values = [None, None]
        df_dict = {"a": values}

        df = pl.DataFrame(df_dict)

        df_nw = nw.from_native(df)

        transformer = ArbitraryImputer(impute_value=impute_value, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        df_transformed_native = transformer.transform(df_nw.to_native())

        df_transformed_nw = nw.from_native(df_transformed_native)

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

        u.assert_frame_equal_dispatch(expected.to_native(), df_transformed_native)

    # have to overload this one, as has slightly different categorical type handling
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_3", "impute_values_dict"),
        [
            ("pandas", "pandas", {"b": "g", "c": "f"}),
            ("polars", "polars", {"b": "g", "c": "f"}),
        ],
        indirect=["expected_df_3"],
    )
    def test_expected_output_with_object_and_categorical_columns(
        self,
        library,
        expected_df_3,
        initialized_transformers,
        impute_values_dict,
        from_json,
    ):
        """Test that transform is giving the expected output when applied to object and categorical columns."""
        # Create the DataFrame using the library parameter
        df2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        transformer.impute_values_ = impute_values_dict

        if self.transformer_name == "ArbitraryImputer":
            transformer.impute_value = "f"

        transformer.columns = ["b", "c"]

        transformer = _handle_from_json(transformer, from_json)

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        u.assert_frame_equal_dispatch(
            df_transformed,
            expected_df_3,
            # this turns off checks for category metadata like ordering
            # this transformer will convert an unordered pd categorical to ordered
            # so this is needed
            check_categorical=False,
        )
        df2 = nw.from_native(df2)
        expected_df_3 = nw.from_native(expected_df_3)

        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_3[[i]].to_native()

            u.assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
                # this turns off checks for category metadata like ordering
                # this transformer will convert an unordered pd categorical to ordered
                # so this is needed
                check_categorical=False,
            )

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("library", "expected_df_4", "impute_values_dict"),
        [
            ("pandas", "pandas", {"b": "z", "c": "z"}),
            ("polars", "polars", {"b": "z", "c": "z"}),
        ],
        indirect=["expected_df_4"],
    )
    def test_expected_output_when_adding_new_categorical_level(
        self,
        library,
        expected_df_4,
        initialized_transformers,
        impute_values_dict,
        from_json,
    ):
        """Test that transform is giving the expected output when applied to object and categorical columns
        (when we're imputing with a new categorical level, which is only possible for arbitrary imputer).
        """
        # Create the DataFrame using the library parameter
        df2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]

        transformer.impute_values_ = impute_values_dict
        transformer.impute_value = "z"
        transformer.columns = ["b", "c"]

        transformer = _handle_from_json(transformer, from_json)

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        u.assert_frame_equal_dispatch(
            df_transformed,
            expected_df_4,
            # this turns off checks for category metadata like ordering
            # this transformer will convert an unordered pd categorical to ordered
            # so this is needed
            check_categorical=False,
        )
        df2 = nw.from_native(df2)
        expected_df_4 = nw.from_native(expected_df_4)

        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_4[[i]].to_native()

            u.assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
                # this turns off checks for category metadata like ordering
                # this transformer will convert an unordered pd categorical to ordered
                # so this is needed
                check_categorical=False,
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
        from_json,
    ):
        """Test that unexpected dtypes will hit error"""

        column = "a"
        df_dict = {column: input_values}

        # because of weird types, initialise manually
        df = pd.DataFrame(df_dict) if library == "pandas" else pl.DataFrame(df_dict)

        transformer = ArbitraryImputer(impute_value=1, columns=[column])

        transformer = _handle_from_json(transformer, from_json)

        bad_types = dict(nw.from_native(df).select(nw.col(column)).schema.items())

        msg = re.escape(
            f"""
                {self.transformer_name}: transformer can only handle Float/Int/Boolean/String/Categorical/Unknown type columns
                but got columns with types {bad_types}
                """,
        )

        with pytest.raises(
            TypeError,
            match=msg,
        ):
            transformer.transform(df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"
