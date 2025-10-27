import datetime

import numpy as np
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    ReturnNativeTests,
    TwoColumnListInitTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
)
from tests.utils import (
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
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
            BeartypeCallHintParamViolation,
        ):
            DateDifferenceTransformer(
                columns=["dummy_1", "dummy_2"],
                new_column_name="dummy_3",
                units="y",
                verbose=False,
            )


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


@pytest.fixture()
def generic_expected_df():
    def _expected_df(unit, library="pandas"):
        a = [
            datetime.datetime(1993, 9, 27, 11, 58, 58, tzinfo=datetime.timezone.utc),
            datetime.datetime(2000, 3, 19, 12, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 10, 10, 10, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 12, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(1985, 7, 23, 11, 59, 59, tzinfo=datetime.timezone.utc),
        ]
        b = [
            datetime.datetime(2020, 5, 1, 12, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2019, 12, 25, 11, 58, 58, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 11, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2018, 9, 10, 9, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, 11, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 11, 10, 12, 59, 59, tzinfo=datetime.timezone.utc),
            datetime.datetime(2015, 7, 23, 11, 59, 59, tzinfo=datetime.timezone.utc),
        ]
        values = {
            "D": [9713.0, 7220.0, 0.0, 31.0, -30.0, -1065.0, -1126.0, 10957.0],
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
            "custom_days": [388.52, 288.80, 0.0, 1.24, -1.20, -42.60, -45.04, 438.28],
        }
        df_dict = {
            "a": a,
            "b": b,
            unit: values[unit],
        }
        return dataframe_init_dispatch(df_dict, library=library)

    return _expected_df


class TestTransform(
    GenericTransformTests,
    GenericDatesMixinTransformTests,
    DropOriginalTransformMixinTests,
    ReturnNativeTests,
):
    """Tests for DateDifferenceTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDifferenceTransformer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "unit",
        [
            "D",
            "h",
            "m",
            "s",
            "week",
            "fortnight",
            "lunar_month",
            "common_year",
            "custom_days",
        ],
    )
    @pytest.mark.parametrize(
        "library",
        [
            "pandas",
            "polars",
        ],
    )
    def test_expected_output_units(self, generic_expected_df, unit, library, from_json):
        """Test that the output is as expected from transform, when units are D, h, m, s, week, fortnight, lunar_month, common_year, or custom_days.

        This tests positive month gaps, negative month gaps, and missing values.
        """
        df = d.create_datediff_test_df(library=library)
        expected = generic_expected_df(unit, library)

        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name=unit,
            units=unit,
            verbose=False,
            custom_days_divider=25 if unit == "custom_days" else None,
        )

        x = _handle_from_json(x, from_json)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize("from_json", [True, False])
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
    def test_expected_output_nulls(self, df, expected, from_json):
        """Test that the output is expected from transform, when columns have nulls."""
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        x = _handle_from_json(x, from_json)

        df_transformed = x.transform(df)

        assert_frame_equal_dispatch(df_transformed, expected)

    @pytest.mark.parametrize("from_json", [True, False])
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
    def test_expected_output_nulls2(self, df, expected, from_json):
        """Test that the output is expected from transform, when columns are nulls."""
        x = DateDifferenceTransformer(
            columns=["a", "b"],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        x = _handle_from_json(x, from_json)

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
