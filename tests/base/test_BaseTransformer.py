import re

import narwhals as nw
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
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
from tubular.base import BaseTransformer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestTransform(GenericTransformTests, ReturnNativeTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize(
        "return_native",
        [True, False],
    )
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_X_returned(
        self,
        minimal_dataframe_lookup,
        uninitialized_transformers,
        minimal_attribute_dict,
        return_native,
        lazy,
        from_json,
    ):
        """Test that X is returned from transform."""
        df = minimal_dataframe_lookup[self.transformer_name]
        args = minimal_attribute_dict[self.transformer_name]
        args["return_native"] = return_native
        x = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(x, df, lazy=lazy, from_json=from_json):
            return

        df = nw.from_native(df)
        expected = df.clone()

        df = nw.to_native(df)
        expected = nw.to_native(expected)

        x = _handle_from_json(x, from_json)

        df_transformed = x.transform(X=_convert_to_lazy(df, lazy))

        if not x.return_native:
            df_transformed = nw.to_native(df_transformed)

        assert_frame_equal_dispatch(
            expected,
            _collect_frame(df_transformed, lazy),
        )

    def test_transform_raises_if_is_fitted_attribute_missing(self):
        """Test that check_is_fitted rejects an object missing is_fitted_."""

        df = d.create_df_1(library="pandas")
        transformer = BaseTransformer(columns=["a"]).fit(df)
        del transformer.is_fitted_

        with pytest.raises(NotFittedError):
            transformer.transform(df)

    def test_is_fitted_attribute_true_after_init(self):
        """Test that is_fitted_ is set to True for non-fitting transformers."""

        df = d.create_df_1(library="pandas")
        transformer = BaseTransformer(columns=["a"]).fit(df)

        assert transformer.is_fitted_ is True
        transformer.transform(df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"

    def test_pipeline_raises_not_fitted_error_when_unfitted(self):
        """Test that a sklearn Pipeline with BaseTransformer works when fitted, but raises NotFittedError when is_fitted_ is deleted."""
        df = d.create_df_1(library="pandas")
        transformer = BaseTransformer(columns=["a"])

        pipeline = Pipeline([("base_transformer", transformer)])

        # Fit the pipeline
        pipeline.fit(df)

        # Transform should work
        result = pipeline.transform(df)
        assert result is not None

        # Delete is_fitted_ from the transformer
        del transformer.is_fitted_

        # Now transform should raise NotFittedError
        with pytest.raises(
            NotFittedError, match=re.escape("Pipeline is not fitted yet.")
        ):
            pipeline.transform(df)

    def test_pipeline_raises_not_fitted_error_when_not_fitted(self):
        """Test that a sklearn Pipeline with BaseTransformer raises NotFittedError when transform is called without fitting."""
        df = d.create_df_1(library="pandas")
        transformer = BaseTransformer(columns=["a"])
        del transformer.is_fitted_  # Simulate not fitted

        pipeline = Pipeline([("base_transformer", transformer)])

        # Transform without fitting should raise NotFittedError
        with pytest.raises(
            NotFittedError, match=re.escape("Pipeline is not fitted yet.")
        ):
            pipeline.transform(df)
