import narwhals as nw
import pytest

import tests.test_data as d
import tests.utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tubular.imputers import ArbitraryImputer


# Dataframe used exclusively in this testing script
@pytest.fixture(params=["pandas", "polars"])
def downcast_df(request):
    """
    Fixture that returns a DataFrame with columns suitable for downcasting
    for both pandas and polars.
    """
    data = {
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    library = request.param

    return u.dataframe_init_dispatch(data, library)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    # overload some inherited arg tests that have been replaced by beartype
    def test_columns_non_string_or_list_error(self):
        pass

    def test_columns_list_element_error(self):
        pass

    def test_verbose_non_bool_error(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"


class TestTransform(GenericImputerTransformTests, GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    def test_impute_value_preserve_dtype(self, downcast_df):
        """Test that downcast dtypes are preserved after imputation."""

        df_nw = nw.from_native(downcast_df)

        df_nw = df_nw.with_columns(
            nw.col("a").cast(nw.Int8),
            nw.col("b").cast(nw.Float32),
        )

        x = ArbitraryImputer(impute_value=1, columns=["a", "b"])
        df_transformed_native = x.transform(df_nw.to_native())

        df_transformed_nw = nw.from_native(df_transformed_native)

        assert df_transformed_nw["a"].dtype == df_nw["a"].dtype
        assert df_transformed_nw["b"].dtype == df_nw["b"].dtype

    @pytest.mark.parametrize(
        ("library", "expected_df_4", "impute_values_dict"),
        [
            ("pandas", "pandas", {"b": "z", "c": "z"}),
            ("polars", "polars", {"b": "z", "c": "z"}),
        ],
        indirect=["expected_df_4"],
    )
    def test_expected_output_4(
        self,
        library,
        expected_df_4,
        initialized_transformers,
        impute_values_dict,
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

        # Transform the DataFrame
        df_transformed = transformer.transform(df2)

        # Check whole dataframes
        u.assert_frame_equal_dispatch(
            df_transformed,
            expected_df_4,
        )
        df2 = nw.from_native(df2)
        expected_df_4 = nw.from_native(expected_df_4)

        # Check outcomes for single rows
        # turn off type change errors to avoid having to type the single rows
        transformer.error_on_type_change = False
        for i in range(len(df2)):
            df_transformed_row = transformer.transform(df2[[i]].to_native())
            df_expected_row = expected_df_4[[i]].to_native()

            u.assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"
