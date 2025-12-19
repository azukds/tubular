import pandas as pd
import pytest

from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerFitTests,
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import ScalingTransformer


class TestInit(BaseNumericTransformerInitTests):
    """Tests for ScalingTransformer.__init__()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ScalingTransformer"
        cls.scaler_type = "min_max"

    def test_invalid_scaler_type(self):
        """Test that an exception is raised for an invalid scaler type."""
        with pytest.raises(
            ValueError,
            match=r"ScalingTransformer: scaler_type should be one of; \['min_max', 'max_abs', 'standard'\]",
        ):
            ScalingTransformer(columns=["a"], scaler_type="invalid_scaler")


class TestFit(BaseNumericTransformerFitTests):
    """Tests for ScalingTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ScalingTransformer"
        cls.scaler_type = "standard"


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for ScalingTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ScalingTransformer"
        cls.scaler_type = "min_max"

    def test_min_max_scaling(self):
        """Test min-max scaling works correctly."""
        df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        transformer = ScalingTransformer(columns=["a", "b"], scaler_type="min_max")
        transformer.fit(df)  # transformer is fitted before transform
        transformed_df = transformer.transform(df)

        expected_df = pd.DataFrame({"a": [0, 0.5, 1], "b": [0, 0.5, 1]})
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_standard_scaling(self):
        """Test standard scaling works correctly."""
        df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        transformer = ScalingTransformer(columns=["a", "b"], scaler_type="standard")
        transformer.fit(df)
        transformed_df = transformer.transform(df)

        expected_df = pd.DataFrame(
            {
                "a": [-1.22474487, 0, 1.22474487],  # Standardized values
                "b": [-1.22474487, 0, 1.22474487],
            },
        )
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_max_abs_scaling(self):
        """Test max absolute scaling works correctly."""
        df = pd.DataFrame({"a": [-3, -2, -1], "b": [1, 2, 3]})
        transformer = ScalingTransformer(columns=["a", "b"], scaler_type="max_abs")
        transformer.fit(df)
        transformed_df = transformer.transform(df)

        expected_df = pd.DataFrame({"a": [-1, -2 / 3, -1 / 3], "b": [1 / 3, 2 / 3, 1]})
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_transform_min_max_raises(self):
        """Test that transform scales negative values correctly with MinMaxScaler."""
        df = pd.DataFrame({"a": [-3, -2, -1]})  # Example for Min-Max scalingcode
        transformer = ScalingTransformer(columns=["a"], scaler_type="min_max")
        transformer.fit(df)

        transformed_df = transformer.transform(df)

        # Since MinMaxScaler scales the data to [0, 1] based on the min and max of the input,
        # we calculate the expected scaled values for column 'a'
        expected_df = pd.DataFrame({"a": [0, 0.5, 1]})

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_custom_scaler_args(self):
        """Test that custom arguments passed to the scaler are correctly applied."""
        df = pd.DataFrame({"a": [0, 1, 2]})
        transformer = ScalingTransformer(
            columns=["a"],
            scaler_type="min_max",
            scaler_kwargs={"feature_range": (0, 2)},
        )
        transformer.fit(df)
        transformed_df = transformer.transform(df)

        expected_df = pd.DataFrame({"a": [0, 1, 2]}, dtype=float)
        pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=True)
