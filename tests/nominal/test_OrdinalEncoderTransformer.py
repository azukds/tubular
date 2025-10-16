import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.utils import assert_frame_equal_dispatch
from tubular.mapping import BaseMappingTransformer
from tubular.nominal import OrdinalEncoderTransformer


# Dataframe used exclusively in this testing script
def create_OrdinalEncoderTransformer_test_df():
    """Create DataFrame to use OrdinalEncoderTransformer tests that correct values are.

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["a", "b", "c", "d", "e", "f"],
            "c": ["a", "b", "c", "d", "e", "f"],
            "d": [1, 2, 3, 4, 5, 6],
            "e": [3, 4, 5, 6, 7, 8.0],
            "f": [False, False, False, True, True, True],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for OrdinalEncoderTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OrdinalEncoderTransformer"


class TestFit(GenericFitTests, WeightColumnFitMixinTests):
    """Tests for OrdinalEncoderTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OrdinalEncoderTransformer"

    def test_learnt_values(self):
        """Test that the ordinal encoder values learnt during fit are expected."""
        df = create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(columns=["b", "d", "f"])

        x.fit(df, df["a"])

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                    "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                    "f": {False: 1, True: 2},
                },
            },
            msg="mappings attribute",
        )

    def test_learnt_values_weight(self):
        """Test that the ordinal encoder values learnt during fit are expected if a weights column is specified."""
        df = create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(weights_column="e", columns=["b", "d", "f"])

        x.fit(df, df["a"])

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                    "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                    "f": {False: 1, True: 2},
                },
            },
            msg="mappings attribute",
        )

    def test_response_column_nulls_error(self):
        """Test that an exception is raised if nulls are present in response_column."""
        df = d.create_df_4()

        x = OrdinalEncoderTransformer(columns=["b"])

        with pytest.raises(
            ValueError,
            match="OrdinalEncoderTransformer: y has 1 null values",
        ):
            x.fit(df, df["a"])

    def test_error_for_too_many_levels(self):
        "test that transformer.transform errors for column with too many levels"
        transformer = OrdinalEncoderTransformer(columns=["a"])

        df = pd.DataFrame(
            {
                "a": list(range(1000)),
                "b": list(range(1000)),
            },
        )

        with pytest.raises(
            ValueError,
            match="OrdinalEncoderTransformer: column a has too many levels to encode",
        ):
            transformer.fit(df, df["b"])


class TestTransform(GenericTransformTests):
    """Tests for OrdinalEncoderTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OrdinalEncoderTransformer"

    def expected_df_1():
        """Expected output for ."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": ["a", "b", "c", "d", "e", "f"],
                "d": [1, 2, 3, 4, 5, 6],
                "e": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [1, 1, 1, 2, 2, 2],
            },
        )

        df["c"] = df["c"].astype("category")
        for col in ["b", "d", "f"]:
            df[col] = df[col].astype(np.int8)

        return df

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in transform."""
        df = create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(columns="b")

        x.fit(df, df["a"])

        x2 = OrdinalEncoderTransformer(columns="b")

        x2.fit(df, df["a"])

        x2.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Mean response values not changed in transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_OrdinalEncoderTransformer_test_df(),
            expected_df_1(),
        ),
    )
    def test_expected_output(self, df, expected):
        """Test that the output is expected from transform."""
        x = OrdinalEncoderTransformer(columns=["b", "d", "f"])

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        mappings = {
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
            "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            "f": {False: 1, True: 2},
        }

        # mimic fit logic and use base  class to set attrs
        # use BaseMappingTransformer init to process args
        # extract null_mappings from mappings etc
        base_mapping_transformer = BaseMappingTransformer(
            mappings=mappings,
        )

        x.mappings = base_mapping_transformer.mappings
        x.mappings_from_null = base_mapping_transformer.mappings_from_null
        x.return_dtypes = dict.fromkeys(x.columns, "Int8")

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for OrdinalEncoderTransformer outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OrdinalEncoderTransformer"
