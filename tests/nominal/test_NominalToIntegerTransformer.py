import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from test_BaseNominalTransformer import GenericNominalTransformTests

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)
from tubular.mapping import BaseMappingTransformer
from tubular.nominal import NominalToIntegerTransformer


class TestInit(ColumnStrListInitTests):
    """Tests for NominalToIntegerTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NominalToIntegerTransformer"

    def test_start_encoding_not_int_error(self):
        """Test that an exception is raised if start_encoding is not an int."""
        with pytest.raises(ValueError):
            NominalToIntegerTransformer(columns="a", start_encoding="a")


class TestFit(GenericFitTests):
    """Tests for NominalToIntegerTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NominalToIntegerTransformer"

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"], start_encoding=1)

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "a": {k: i for i, k in enumerate(df["a"].unique(), 1)},
                    "b": {k: i for i, k in enumerate(df["b"].unique(), 1)},
                },
            },
            msg="mappings attribute",
        )

    def test_error_for_too_many_levels(self):
        "test that transformer.transform errors for column with too many levels"
        transformer = NominalToIntegerTransformer(columns=["a"])

        df = pd.DataFrame(
            {
                "a": list(range(1000)),
                "b": list(range(1000)),
            },
        )

        with pytest.raises(
            ValueError,
            match="NominalToIntegerTransformer: column a has too many levels to encode",
        ):
            transformer.fit(df, df["b"])


class TestTransform(GenericNominalTransformTests):
    """Tests for NominalToIntegerTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NominalToIntegerTransformer"

    def expected_df_1():
        """Expected output for test_expected_output."""
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]},
        )

        df["a"] = (
            df["a"]
            .replace({k: i for i, k in enumerate(df["a"].unique())})
            .astype(np.int8)
        )

        df["b"] = (
            df["b"]
            .replace({k: i for i, k in enumerate(df["b"].unique())})
            .astype(np.int8)
        )

        return df

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in transform."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        x2 = NominalToIntegerTransformer(columns=["a", "b"])

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Impute values not changed in transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test that the output is expected from transform."""
        x = NominalToIntegerTransformer(columns=["a", "b"])

        # set the mapping dict directly rather than fitting x on df so test works with helpers
        mappings = {
            "a": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
            "b": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5},
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

        x.null_mappings = {"a": None, "b": None}

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in NominalToIntegerTransformer.transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NominalToIntegerTransformer"
