"""Note, BaseNominalTransformer itself has now been removed, 
but it is still useful to have these inheritable tests."""
import copy

import narwhals as nw
import pytest
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    GenericTransformTests,
)
from tests.utils import (
    _check_if_skip_test,
    _handle_from_json,
    assert_frame_equal_dispatch,
)

class GenericNominalTransformTests(GenericTransformTests):
    """
    Tests for nominal module transform methods
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_not_fitted_error_raised(self, initialized_transformers, library):
        if initialized_transformers[self.transformer_name].FITS:
            df = d.create_df_1(library=library)

            transformer = initialized_transformers[self.transformer_name]

            if _check_if_skip_test(transformer, df, lazy=False, from_json=False):
                return

            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(df)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_updated(
        self,
        initialized_transformers,
        library,
        from_json,
    ):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_1(library=library)

        transformer = initialized_transformers[self.transformer_name]

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = transformer.fit(df)

        transformer.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        transformer = _handle_from_json(transformer, from_json)

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, d.create_df_1(library=library))

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_empty_in_empty_out(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
        from_json,
    ):
        """Test transforming empty frame returns empty frame"""

        df = minimal_dataframe_lookup[self.transformer_name]
        x = initialized_transformers[self.transformer_name]

        if _check_if_skip_test(x, df, lazy=False, from_json=from_json):
            return

        x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        x = _handle_from_json(x, from_json)

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

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas"],
    )
    def test_pandas_index_not_updated(
        self,
        initialized_transformers,
        library,
        from_json,
    ):
        """Test that the original (pandas) dataframe index is not transformed when transform method used."""

        df = d.create_df_1(library=library)

        x = initialized_transformers[self.transformer_name]

        if _check_if_skip_test(x, df, lazy=False, from_json=from_json):
            return

        x = x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        x = _handle_from_json(x, from_json)

        # update to abnormal index
        df.index = [2 * i for i in df.index]

        original_df = copy.deepcopy(df)

        _ = x.transform(df)

        assert_frame_equal_dispatch(df, original_df)