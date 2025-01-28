import copy
import re

import pandas as pd
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.mapping import BaseMappingTransformMixin

# Note there are no tests that need inheriting from this file as the only difference is an expected transform output


@pytest.fixture()
def mapping():
    return {
        "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: None},
        "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, None: 9},
    }


class TestInit(ColumnStrListInitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestTransform(GenericTransformTests):
    """
    Tests for BaseMappingTransformMixin.transform().

    Because this is a Mixin transformer it is not always appropriate to inherit the generic transform tests. A number of the tests below overwrite the tests in GenericTransformTests.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output(self, mapping, library):
        """Test that X is returned from transform."""

        df = d.create_df_1(library=library)

        expected_dict = {
            "a": ["a", "b", "c", "d", "e", "f"],
            "b": [1, 2, 3, 4, 5, 6],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        transformer = BaseMappingTransformMixin(columns=["a", "b"])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {"a": "String", "b": "Int64"}

        df_transformed = transformer.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_mappings_unchanged(self, mapping, library):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_1(library=library)

        transformer = BaseMappingTransformMixin(columns=["a", "b"])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {
            "a": "String",
            "b": "Int64",
        }

        transformer.transform(df)

        assert (
            mapping == transformer.mappings
        ), f"BaseMappingTransformer.transform has changed self.mappings unexpectedly, expected {mapping} but got {transformer.mappings}"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    def test_non_pd_type_error(
        self,
        non_df,
        mapping,
        library,
    ):
        """Test that an error is raised in transform is X is not a pd.DataFrame."""

        df = d.create_df_10(library=library)

        transformer = BaseMappingTransformMixin(columns=["a"])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {
            "a": "String",
        }

        x_fitted = transformer.fit(df, df["c"])

        with pytest.raises(
            TypeError,
            match="BaseMappingTransformMixin: X should be a polars or pandas DataFrame/LazyFrame",
        ):
            x_fitted.transform(X=non_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_no_rows_error(self, mapping, library):
        """Test an error is raised if X has no rows."""
        df = d.create_df_10(library=library)

        transformer = BaseMappingTransformMixin(columns=["a"])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {"a": "String"}

        transformer = transformer.fit(df, df["c"])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape("BaseMappingTransformMixin: X has no rows; (0, 3)"),
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_updated(self, mapping, library):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_10(library=library)

        transformer = BaseMappingTransformMixin(columns=["a"])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {"a": "String", "b": "Int64"}

        transformer = transformer.fit(df, df["c"])

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, d.create_df_10(library=library))

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas"],
        indirect=True,
    )
    def test_pandas_index_not_updated(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
        mapping,
    ):
        """Test that the original (pandas) dataframe index is not transformed when transform method used."""

        df = minimal_dataframe_lookup[self.transformer_name]
        transformer = initialized_transformers[self.transformer_name]

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        transformer.mappings = mapping
        transformer.return_dtypes = {"a": "String", "b": "String"}

        # update to abnormal index
        df.index = [2 * i for i in df.index]

        original_df = copy.deepcopy(df)

        transformer = transformer.fit(df, df["a"])

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, original_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"
