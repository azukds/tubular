import datetime

import numpy as np
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.dates import DateDifferenceTransformer


class TestInit(
    TwoColumnListInitTests,
    DropOriginalInitMixinTests,
    NewColumnNameInitMixintests,
):
    """Tests for DateDifferenceTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDifferenceTransformer"

    def test_units_values_error(self):
        """Test that an exception is raised if the value of inits is not one of accepted_values_units."""
        with pytest.raises(
            ValueError,
            match=r"DateDifferenceTransformer: units must be one of \['week', 'fortnight', 'lunar_month', 'common_year', 'custom_days', 'D', 'h', 'm', 's'\], got y",
        ):
            DateDifferenceTransformer(
                columns=["dummy_1", "dummy_2"],
                new_column_name="dummy_3",
                units="y",
                verbose=False,
            )


def expected_df_3(library="pandas"):
    """Expected output for test_expected_output_units_D."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "D": [
            9713.0,
            7220.0,
            0.0,
            31.0,
            -30.0,
            -1065.0,
            -1126.0,
            10957.0,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_4(library="pandas"):
    """Expected output for test_expected_output_units_h."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "h": [
            233113.01694444445,
            173278.98305555555,
            0.0,
            744.0,
            -722.0,
            -25559.0,
            -27023.0,
            262968.0,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_5(library="pandas"):
    """Expected output for test_expected_output_units_m."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "m": [
            13986781.016666668,
            10396738.983333332,
            0.0,
            44640.0,
            -43320.0,
            -1533540.0,
            -1621380.0,
            15778080.0,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_6(library="pandas"):
    """Expected output for test_expected_output_units_s."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "s": [
            839206861.0,
            623804339.0,
            0.0,
            2678400.0,
            -2599200.0,
            -92012400.0,
            -97282800.0,
            946684800.0,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_7(library="pandas"):
    """Expected output for test_expected_output_nulls."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            np.nan if library == "pandas" else None,
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            np.nan if library == "pandas" else None,
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "D": [
            np.nan if library == "pandas" else None,
            np.nan if library == "pandas" else None,
            10957.0,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def create_datediff_test_nulls_df(library="pandas"):
    """Create DataFrame with nulls for DateDifferenceTransformer tests."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            np.nan if library == "pandas" else None,
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            np.nan if library == "pandas" else None,
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_8(library="pandas"):
    """Expected output for test_expected_output_nulls2."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            np.nan if library == "pandas" else None,
        ],
        "b": [
            np.nan if library == "pandas" else None,
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "D": [
            np.nan if library == "pandas" else None,
            np.nan if library == "pandas" else None,
        ],
    }
    if library == "polars":
        expected = dataframe_init_dispatch(df_dict, library=library)
        # change the D column to float as in the transformer it is a calculated field and this is automatically set up as float.
        return expected.cast({"D": pl.Float64})
    return dataframe_init_dispatch(df_dict, library=library)


def create_datediff_test_nulls_df2(library="pandas"):
    """Create DataFrame with nulls only for DateDifferenceTransformer tests."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            np.nan if library == "pandas" else None,
        ],
        "b": [
            np.nan if library == "pandas" else None,
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_9(library="pandas"):
    """Expected output for test_expected_output_units_week."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "week": [
            1387.571429,
            1031.428571,
            0.0,
            4.428571,
            -4.285714,
            -152.142857,
            -160.857143,
            1565.285714,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_10(library="pandas"):
    """Expected output for test_expected_output_units_fortnight."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "fortnight": [
            693.785714,
            515.714286,
            0.0,
            2.214286,
            -2.142857,
            -76.071429,
            -80.428571,
            782.642857,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_11(library="pandas"):
    """Expected output for test_expected_output_units_lunar_month."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "lunar_month": [
            329.254237,
            244.745763,
            0.0,
            1.050847,
            -1.016949,
            -36.101695,
            -38.169492,
            371.423729,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_12(library="pandas"):
    """Expected output for test_expected_output_units_common_year."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "common_year": [
            26.610958,
            19.780822,
            0.0,
            0.084932,
            -0.082192,
            -2.917808,
            -3.084932,
            30.019178,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


def expected_df_13(library="pandas"):
    """Expected output for test_expected_output_units_custom_days."""

    df_dict = {
        "a": [
            datetime.datetime(
                1993,
                9,
                27,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2000,
                3,
                19,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                10,
                10,
                10,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                12,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                1985,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "b": [
            datetime.datetime(
                2020,
                5,
                1,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2019,
                12,
                25,
                11,
                58,
                58,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2018,
                9,
                10,
                9,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                11,
                10,
                12,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
            datetime.datetime(
                2015,
                7,
                23,
                11,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ),
        ],
        "custom_days": [
            388.52,
            288.80,
            0.0,
            1.24,
            -1.20,
            -42.60,
            -45.04,
            438.28,
        ],
    }

    return dataframe_init_dispatch(df_dict, library=library)


class TestTransform(
    GenericTransformTests,
    GenericDatesMixinTransformTests,
    DropOriginalTransformMixinTests,
):
    """Tests for DateDifferenceTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDifferenceTransformer"

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_3(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_3(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_D(self, df, expected):
        """Test that the output is expected from transform, when units is D.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_4(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_4(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_h(self, df, expected):
        """Test that the output is expected from transform, when units is h.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="h",
            units="h",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_5(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_5(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_m(self, df, expected):
        """Test that the output is expected from transform, when units is m.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="m",
            units="m",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_6(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_6(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_s(self, df, expected):
        """Test that the output is expected from transform, when units is s.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="s",
            units="s",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_9(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_9(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_week(self, df, expected):
        """Test that the output is expected from transform, when units is week.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="week",
            units="week",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_10(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_10(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_fortnight(self, df, expected):
        """Test that the output is expected from transform, when units is fortnight.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="fortnight",
            units="fortnight",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_11(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_11(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_lunar_month(self, df, expected):
        """Test that the output is expected from transform, when units is lunar_month.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="lunar_month",
            units="lunar_month",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_12(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_12(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_common_year(self, df, expected):
        """Test that the output is expected from transform, when units is common_year.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="common_year",
            units="common_year",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                d.create_datediff_test_df(library="pandas"),
                expected_df_13(library="pandas"),
            ),
            (
                d.create_datediff_test_df(library="polars"),
                expected_df_13(library="polars"),
            ),
        ],
    )
    def test_expected_output_units_custom_days(self, df, expected):
        """Test that the output is expected from transform, when units is custom_days.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="custom_days",
            units="custom_days",
            custom_days_divider=25,
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_datediff_test_nulls_df(library="pandas"),
                expected_df_7(library="pandas"),
            ),
            (
                create_datediff_test_nulls_df(library="polars"),
                expected_df_7(library="polars"),
            ),
        ],
    )
    def test_expected_output_nulls(self, df, expected):
        """Test that the output is expected from transform, when columns have nulls."""
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_datediff_test_nulls_df2(library="pandas"),
                expected_df_8(library="pandas"),
            ),
            (
                create_datediff_test_nulls_df2(library="polars"),
                expected_df_8(library="polars"),
            ),
        ],
    )
    def test_expected_output_nulls2(self, df, expected):
        """Test that the output is expected from transform, when columns are nulls."""
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDifferenceTransformer"
