import copy
import datetime

import narwhals as nw
import numpy as np
import polars as pl
import pytest
from dateutil.tz import gettz

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
)
from tests.test_data import create_date_diff_different_dtypes, create_date_test_df
from tests.utils import _check_if_skip_test, _handle_from_json, dataframe_init_dispatch
from tubular.dates import TIME_UNITS


class GenericDatesMixinTransformTests:
    """Generic tests for Dates Transformers"""

    @pytest.mark.parametrize("from_json", [True, False])
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
        from_json,
    ):
        "Test that transform raises an error if columns contains non date types"

        args = minimal_attribute_dict[self.transformer_name].copy()
        columns = args["columns"]

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = copy.deepcopy(minimal_dataframe_lookup[self.transformer_name])

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

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

    @pytest.mark.parametrize("from_json", [True, False])
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
        from_json,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = create_date_diff_different_dtypes(library=library)

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df = (
            nw.from_native(df)
            .with_columns(
                nw.col(col).cast(nw.Datetime(time_unit="ns", time_zone="UTC"))
                for col in ["datetime_col_1", "datetime_col_2"]
            )
            .to_native()
        )

        present_types = (
            {nw.Datetime, nw.Date()} if datetime_col == 0 else {nw.Date(), nw.Datetime}
        )
        # convert to list and sort to ensure reproducible order
        present_types = {str(value) for value in present_types}
        present_types = list(present_types)
        present_types.sort()
        msg = f"Columns fed to datetime transformers should be ['Datetime', 'Date'] and have consistent types, but found {present_types}. Note, Datetime columns should have time_unit in {TIME_UNITS} and time_zones from zoneinfo.available_timezones(). Please use ToDatetimeTransformer to standardise."

        with pytest.raises(
            TypeError,
        ) as exc_info:
            transformer.transform(df)

        assert msg in str(exc_info.value)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas"])
    @pytest.mark.parametrize(
        "bad_timezone",
        [
            "Factory",
            "localtime",
        ],
    )
    def test_bad_timezones_error(
        self,
        bad_timezone,
        uninitialized_transformers,
        minimal_attribute_dict,
        library,
        from_json,
    ):
        """Test that transform raises an error if
        datetime columns have non-accepted timezones

        Note:
        - polars outright rejects these at df init, so nothing to test
        - pandas accepts these, but narwhals processes into Unknown type,
        so this still goes through our usual bad dtype error handling
        """
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = ["a", "b"]

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df_dict = {
            "a": [
                datetime.datetime(1993, 9, 27, tzinfo=gettz(bad_timezone)),
                datetime.datetime(2000, 3, 19, tzinfo=gettz(bad_timezone)),
            ],
            "b": [
                datetime.datetime(1993, 9, 27, tzinfo=gettz("UTC")),
                datetime.datetime(2000, 3, 19, tzinfo=gettz("UTC")),
            ],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        msg = "a type should be in ['Datetime', 'Date'] but got Unknown. Note, Datetime columns should have time_unit in ['us', 'ns', 'ms'] and time_zones from zoneinfo.available_timezones()"

        with pytest.raises(
            TypeError,
        ) as exc_info:
            transformer.transform(df)

        assert msg in str(exc_info.value)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_only_typechecks_self_columns(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        library,
        from_json,
    ):
        "Test that type checks are only performed on self.columns"
        args = minimal_attribute_dict[self.transformer_name].copy()

        transformer = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = create_date_test_df(library=library)

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        df = nw.from_native(df)

        # add non datetime column
        df = df.with_columns(
            nw.new_series(
                name="z",
                values=[
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "h",
                ],
                backend=nw.get_native_namespace(df),
            ),
        ).to_native()

        # if transformer is not yet polars compatible, skip this test
        if not transformer.polars_compatible and isinstance(df, pl.DataFrame):
            return

        # test that this runs successfully
        transformer.transform(df)


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


class TestTransform(
    GenericTransformTests,
    GenericDatesMixinTransformTests,
    ReturnNativeTests,
):
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
