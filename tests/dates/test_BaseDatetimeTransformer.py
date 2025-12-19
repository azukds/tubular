from copy import deepcopy

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.utils import _check_if_skip_test, _convert_to_lazy
from tubular.dates import TIME_UNITS


class DatetimeMixinTransformTests:
    """Generic tests for Datetime Transformers"""

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=["minimal_dataframe_lookup"],
    )
    @pytest.mark.parametrize(
        ("bad_value", "bad_type"),
        [
            (1, "Int64"),
            ("a", "String"),
            (np.nan, "Float64"),
            (pd.to_datetime("01/02/2020").date(), "Date"),
        ],
    )
    def test_non_datetypes_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        minimal_dataframe_lookup,
        bad_value,
        bad_type,
        lazy,
    ):
        "Test that transform raises an error if columns contains non datetime types"

        args = minimal_attribute_dict[self.transformer_name].copy()
        columns = args["columns"]

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = deepcopy(minimal_dataframe_lookup[self.transformer_name])

        if _check_if_skip_test(transformer, df, lazy):
            return

        for i in range(len(columns)):
            col = columns[i]
            # force date test to wanted type if pandas df, to avoid
            # narwhals NotImplementedError where Date dtype is only
            # supported for pyarrow-backed dtypes in pandas
            if bad_type == "Date" and isinstance(df, pd.DataFrame):
                bad_df = deepcopy(df)
                bad_df[col] = bad_df[col].astype("date32[pyarrow]")
                bad_df = nw.from_native(bad_df)
            else:
                bad_df = nw.from_native(df).clone()
                bad_df = bad_df.with_columns(
                    nw.lit(bad_value).cast(getattr(nw, bad_type)).alias(col),
                )

            msg = rf"{col} type should be in ['Datetime'] but got {bad_type}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones()"

            with pytest.raises(
                TypeError,
            ) as exc_info:
                transformer.transform(nw.to_native(_convert_to_lazy(bad_df, lazy)))

            assert msg in str(exc_info.value)


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestTransform(
    GenericTransformTests,
    DatetimeMixinTransformTests,
    ReturnNativeTests,
):
    """Tests for BaseDatetimeTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"
