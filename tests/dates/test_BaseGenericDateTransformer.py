import copy

import narwhals as nw
import numpy as np
import polars as pl
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tests.test_data import create_date_diff_different_dtypes
from tubular.dates import TIME_UNITS


class GenericDatesMixinTransformTests:
    """Generic tests for Dates Transformers"""

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=["minimal_dataframe_lookup"],
    )
    @pytest.mark.parametrize(
        ("bad_value", "bad_type"),
        [
            (1, nw.Int64()),
            ("a", nw.String()),
            (np.nan, nw.Float64()),
        ],
    )
    def test_non_datetypes_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        minimal_dataframe_lookup,
        bad_value,
        bad_type,
    ):
        "Test that transform raises an error if columns contains non date types"

        args = minimal_attribute_dict[self.transformer_name].copy()
        columns = args["columns"]

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = copy.deepcopy(minimal_dataframe_lookup[self.transformer_name])

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        for i in range(len(columns)):
            col = columns[i]
            bad_df = nw.from_native(df).clone()
            bad_df = bad_df.with_columns(
                nw.lit(bad_value).cast(bad_type).alias(col),
            )

            msg = rf"{col} type should be in ['Datetime', 'Date'] but got {bad_type}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones()"

            with pytest.raises(
                TypeError,
            ) as exc_info:
                transformer.transform(nw.to_native(bad_df))

            assert msg in str(exc_info.value)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("columns, datetime_col"),
        [
            (["date_col_1", "datetime_col_2"], 1),
            (["datetime_col_1", "date_col_2"], 0),
        ],
    )
    def test_mismatched_datetypes_error(
        self,
        columns,
        datetime_col,
        uninitialized_transformers,
        minimal_attribute_dict,
        library,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = create_date_diff_different_dtypes(library=library)

        df = (
            nw.from_native(df)
            .with_columns(
                nw.col(col).cast(nw.Datetime(time_unit="ns", time_zone="UTC"))
                for col in ["datetime_col_1", "datetime_col_2"]
            )
            .to_native()
        )

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        present_types = (
            {nw.Datetime(time_unit="ns", time_zone="UTC"), nw.Date()}
            if datetime_col == 0
            else {nw.Date(), nw.Datetime(time_unit="ns", time_zone="UTC")}
        )
        msg = f"Columns fed to datetime transformers should be ['Datetime', 'Date'] and have consistent types, but found {present_types}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones(). Please use ToDatetimeTransformer to standardise."

        with pytest.raises(
            TypeError,
        ) as exc_info:
            transformer.transform(df)

        assert msg in str(exc_info.value)


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestTransform(GenericTransformTests, GenericDatesMixinTransformTests):
    """Tests for BaseGenericDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"
