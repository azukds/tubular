import datetime

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.dates import ToDatetimeTransformer


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"

    def test_time_format_type_error(self):
        """Test that an exception is raised for bad time_zone arg."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            ToDatetimeTransformer(column="a", time_format=1)

    def test_warning_for_none_time_format(self):
        "test appropriate warning raised when time_format not provided"

        with pytest.warns(
            UserWarning,
            match="time_format arg has not been provided, so datetime format will be inferred",
        ):
            ToDatetimeTransformer(columns=["a"])


class TestTransform(GenericTransformTests):
    """Tests for BaseDatetimeTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"

    def expected_df_1(self, library="pandas"):
        """Expected output for test_expected_output."""

        df_dict = {
            "a": [
                # ignore the rule that insists on timezones for following,
                # as complicates tests and lack of tz is not meaningful here
                datetime.datetime(1950, 1, 1),  # noqa: DTZ001
                datetime.datetime(1960, 1, 1),  # noqa: DTZ001
                datetime.datetime(2000, 1, 1),  # noqa: DTZ001
                datetime.datetime(2001, 1, 1),  # noqa: DTZ001
                None,
                datetime.datetime(2010, 1, 1),  # noqa: DTZ001
            ],
            "b": [
                datetime.datetime(2001, 1, 1),  # noqa: DTZ001
                None,
                datetime.datetime(2002, 1, 1),  # noqa: DTZ001
                datetime.datetime(2004, 1, 1),  # noqa: DTZ001
                None,
                datetime.datetime(2010, 1, 1),  # noqa: DTZ001
            ],
            "c": [
                datetime.datetime(2025, 2, 1),  # noqa: DTZ001
                datetime.datetime(1996, 4, 3),  # noqa: DTZ001
                datetime.datetime(2023, 12, 3),  # noqa: DTZ001
                datetime.datetime(1980, 8, 20),  # noqa: DTZ001
                None,
                None,
            ],
            "d": [
                datetime.datetime(2020, 5, 3),  # noqa: DTZ001
                datetime.datetime(1990, 10, 2),  # noqa: DTZ001
                datetime.datetime(2004, 11, 5),  # noqa: DTZ001
                None,
                None,
                datetime.datetime(1997, 9, 5),  # noqa: DTZ001
            ],
            "e": [
                datetime.datetime(2020, 1, 1, 0, 0, 0),  # noqa: DTZ001
                datetime.datetime(2021, 2, 2, 1, 1, 1),  # noqa: DTZ001
                datetime.datetime(2022, 3, 3, 2, 2, 2),  # noqa: DTZ001
                datetime.datetime(2023, 4, 4, 3, 3, 3),  # noqa: DTZ001
                datetime.datetime(2024, 5, 5, 4, 4, 4),  # noqa: DTZ001
                None,
            ],
        }

        return dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    def create_to_datetime_test_df(self, library="pandas"):
        """Create DataFrame to be used in the ToDatetimeTransformer tests."""

        df_dict = {
            "a": ["1950", "1960", "2000", "2001", None, "2010"],
            "b": ["2001", None, "2002", "2004", None, "2010"],
            "c": ["01/02/2025", "03/04/1996", "03/12/2023", "20/08/1980", None, None],
            "d": ["03/05/2020", "02/10/1990", "05/11/2004", None, None, "05/09/1997"],
            "e": [
                "01/01/2020 00:00:00",
                "02/02/2021 01:01:01",
                "03/03/2022 02:02:02",
                "04/04/2023 03:03:03",
                "05/05/2024 04:04:04",
                None,
            ],
        }

        return dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    @pytest.mark.parametrize(
        ("columns", "time_format"),
        [
            (["a", "b"], "%Y"),
            (["c", "d"], "%d/%m/%Y"),
            (["e"], None),
        ],
    )
    def test_expected_output_year_parsing(self, library, columns, time_format):
        """Test input data is transformed as expected."""

        df = self.create_to_datetime_test_df(library=library)
        expected = self.expected_df_1(library=library)

        to_dt = ToDatetimeTransformer(
            columns=columns,
            time_format=time_format,
        )
        df_transformed = to_dt.transform(df)

        assert_frame_equal_dispatch(expected[columns], df_transformed[columns])


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"
