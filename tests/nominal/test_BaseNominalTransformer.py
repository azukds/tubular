import copy

import narwhals as nw
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch


# The first part of this file builds out the tests for BaseNominalTransformer so that they can be
# imported into other test files (by not starting the class name with Test)
# The second part actually calls these tests (along with all other require tests) for the BaseNominalTransformer
class GenericNominalTransformTests(GenericTransformTests):
    """
    Tests for BaseNominalTransformer.transform().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_not_fitted_error_raised(self, initialized_transformers, library):
        if initialized_transformers[self.transformer_name].FITS:
            df = d.create_df_1(library=library)

            transformer = initialized_transformers[self.transformer_name]

            # skip polars test if transformer not yet converted for polars
            if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
                return

            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_non_mappable_rows_exception_raised(
        self,
        initialized_transformers,
        library,
    ):
        """Test an exception is raised if non-mappable rows are present in X."""
        df = d.create_df_1(library=library)

        transformer = initialized_transformers[self.transformer_name]

        # skip polars test if transformer not yet converted for polars
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.fit(df)

        transformer.mappings = {
            "a": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
            "b": {"a": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: nulls would be introduced into columns b from levels not present in mapping",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_updated(self, initialized_transformers, library):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_1(library=library)

        transformer = initialized_transformers[self.transformer_name]

        # skip polars test if transformer not yet converted for polars
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer = transformer.fit(df)

        transformer.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, d.create_df_1(library=library))

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_empty_in_empty_out(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
    ):
        """Test transforming empty frame returns empty frame"""

        df = minimal_dataframe_lookup[self.transformer_name]
        x = initialized_transformers[self.transformer_name]

        # skip polars test if transformer not yet converted for polars
        if not x.polars_compatible and isinstance(df, pl.DataFrame):
            return

        x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        df = nw.from_native(df)
        # take 0 rows from df
        df = df.head(0).to_native()

        output = x.transform(
            df,
        )

        output = nw.from_native(output)

        assert output.shape[0] == 0, (
            "expected empty frame transform to return empty frame"
        )

    @pytest.mark.parametrize(
        "library",
        ["pandas"],
    )
    def test_pandas_index_not_updated(
        self,
        initialized_transformers,
        library,
    ):
        """Test that the original (pandas) dataframe index is not transformed when transform method used."""

        df = d.create_df_1(library=library)

        x = initialized_transformers[self.transformer_name]

        x = x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        # update to abnormal index
        df.index = [2 * i for i in df.index]

        original_df = copy.deepcopy(df)

        _ = x.transform(df)

        assert_frame_equal_dispatch(df, original_df)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestTransform(GenericNominalTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    # overload test as class needs special  handling to run
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_get_feature_names_out_matches_new_features(
        self,
        library,
        initialized_transformers,
    ):
        """Test that the expected newly created features (if any) are indeed contained
        in the output df"""

        df = d.create_df_1(library=library)

        x = initialized_transformers[self.transformer_name]

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        output = x.transform(df)

        output_columns = set(output.columns)

        expected_new_columns = set(x.get_feature_names_out())

        # are expected columns in the data
        assert expected_new_columns.intersection(output_columns), (
            f"{x.classname()}: get_feature_names_out does not agree with output of .transform, expected {expected_new_columns} but got {output_columns}"
        )

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"
