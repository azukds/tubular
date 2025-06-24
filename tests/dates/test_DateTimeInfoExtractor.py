import datetime

import joblib
import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseDatetimeTransformer import (
    DatetimeMixinTransformTests,
)
from tests.utils import assert_frame_equal_dispatch
from tubular.dates import DatetimeInfoExtractor, DatetimeInfoOptions


@pytest.fixture()
def timeofday_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofday"])


@pytest.fixture()
def timeofmonth_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofmonth"])


@pytest.fixture()
def timeofyear_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofyear"])


@pytest.fixture()
def dayofweek_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["dayofweek"])


class TestInit(
    ColumnStrListInitTests,
):
    "tests for DatetimeInfoExtractor.__init__"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeInfoExtractor"

    @pytest.mark.parametrize("incorrect_type_include", [2, 3.0, "invalid", "dayofweek"])
    def test_error_when_include_not_list(self, incorrect_type_include):
        """Test that an exception is raised when value include variable is not a list."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeInfoExtractor(columns=["a"], include=incorrect_type_include)

    def test_error_when_invalid_include_option(self):
        """Test that an exception is raised when include contains incorrect values."""
        print("invalid_option" in DatetimeInfoOptions._value2member_map_)
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                include=["timeofday", "timeofmonth", "invalid_option"],
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings",
        [2, 3.0, ["a", "b"], "dayofweek"],
    )
    def test_error_when_datetime_mappings_not_dict(
        self,
        incorrect_type_datetime_mappings,
    ):
        """Test that an exception is raised when datetime_mappings is not a dict."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                datetime_mappings=incorrect_type_datetime_mappings,
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings_values",
        [{"timeofday": 2}],
    )
    def test_error_when_datetime_mapping_value_not_dict(
        self,
        incorrect_type_datetime_mappings_values,
    ):
        """Test that an exception is raised when values in datetime_mappings are not dict."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                datetime_mappings=incorrect_type_datetime_mappings_values,
            )

    @pytest.mark.parametrize(
        ("include", "incorrect_datetime_mappings_keys"),
        [
            (["timeofyear"], {"invalid_key": {"valid_mapping": "valid_output"}}),
            (["timeofmonth"], {"bla": {"day": range(7)}}),
        ],
    )
    def test_error_when_datetime_mapping_key_not_allowed(
        self,
        include,
        incorrect_datetime_mappings_keys,
    ):
        """Test that an exception is raised when keys in datetime_mappings are not allowed."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                include=include,
                datetime_mappings=incorrect_datetime_mappings_keys,
            )

    @pytest.mark.parametrize(
        ("include", "incorrect_datetime_mappings_keys"),
        [
            (["timeofyear"], {"dayofweek": {"day": range(7)}}),
            (
                ["timeofyear"],
                {"timeofyear": {"month": range(12)}, "timeofday": {"hour": range(24)}},
            ),
        ],
    )
    def test_error_when_datetime_mapping_key_not_in_include(
        self,
        include,
        incorrect_datetime_mappings_keys,
    ):
        """Test that an exception is raised when keys in datetime_mappings are not in include."""
        with pytest.raises(
            ValueError,
            match="keys in datetime_mappings should be in include",
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                include=include,
                datetime_mappings=incorrect_datetime_mappings_keys,
            )

    @pytest.mark.parametrize(
        ("incomplete_mappings", "expected_exception"),
        [
            (
                {"timeofday": {"mapped": range(23)}},
                r"DatetimeInfoExtractor: timeofday mapping dictionary should contain mapping for all values between 0-23. \{23\} are missing",
            ),
            (
                {"timeofmonth": {"mapped": range(1, 31)}},
                r"DatetimeInfoExtractor: timeofmonth mapping dictionary should contain mapping for all values between 1-31. \{31\} are missing",
            ),
            (
                {"timeofyear": {"mapped": range(1, 12)}},
                r"DatetimeInfoExtractor: timeofyear mapping dictionary should contain mapping for all values between 1-12. \{12\} are missing",
            ),
            (
                {"dayofweek": {"mapped": range(6)}},
                r"DatetimeInfoExtractor: dayofweek mapping dictionary should contain mapping for all values between 1-7. \{6, 7\} are missing",
            ),
        ],
    )
    def test_error_when_incomplete_mappings_passed(
        self,
        incomplete_mappings,
        expected_exception,
    ):
        """Test that error is raised when incomplete mappings are passed."""
        with pytest.raises(ValueError, match=expected_exception):
            DatetimeInfoExtractor(columns=["a"], datetime_mappings=incomplete_mappings)


class TestTransform(
    GenericTransformTests,
    DatetimeMixinTransformTests,
    DropOriginalTransformMixinTests,
):
    "tests for DatetimeInfoExtractor.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeInfoExtractor"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_single_column_output_for_all_options(self, library):
        """Test that correct df is returned after transformation."""
        df = d.create_date_test_df(library=library)
        df = nw.from_native(df)
        backend = nw.get_native_namespace(df)
        df = df.with_columns(
            nw.new_series(
                name="b",
                values=[
                    None,
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        12,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        10,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        9,
                        10,
                        18,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        22,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        19,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        7,
                        23,
                        3,
                        0,
                        0,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                backend=backend,
                dtype=nw.Datetime(time_unit="us", time_zone="UTC"),
            ),
        )

        transformer = DatetimeInfoExtractor(
            columns=["b"],
            include=["timeofmonth", "timeofyear", "dayofweek", "timeofday"],
        )
        transformed = transformer.transform(df.to_native())

        expected = df.clone()
        expected = df.with_columns(
            nw.new_series(
                name="b_timeofmonth",
                values=[
                    None,
                    "end",
                    "start",
                    "start",
                    "start",
                    "start",
                    "start",
                    "end",
                ],
                backend=backend,
                dtype=nw.Enum(["end", "middle", "start"]),
            ),
            nw.new_series(
                name="b_timeofyear",
                values=[
                    None,
                    "winter",
                    "autumn",
                    "autumn",
                    "autumn",
                    "autumn",
                    "autumn",
                    "summer",
                ],
                backend=backend,
                dtype=nw.Enum(["autumn", "spring", "summer", "winter"]),
            ),
            nw.new_series(
                name="b_dayofweek",
                values=[
                    None,
                    "wednesday",
                    "saturday",
                    "saturday",
                    "monday",
                    "tuesday",
                    "tuesday",
                    "thursday",
                ],
                backend=nw.get_native_namespace(df),
                dtype=nw.Enum(
                    [
                        "friday",
                        "monday",
                        "saturday",
                        "sunday",
                        "thursday",
                        "tuesday",
                        "wednesday",
                    ],
                ),
            ),
            nw.new_series(
                name="b_timeofday",
                values=[
                    None,
                    "afternoon",
                    "morning",
                    "morning",
                    "evening",
                    "evening",
                    "evening",
                    "night",
                ],
                backend=nw.get_native_namespace(df),
                dtype=nw.Enum(["afternoon", "evening", "morning", "night"]),
            ),
        )

        assert_frame_equal_dispatch(transformed, expected.to_native())

        # also test single row
        df = nw.from_native(df)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_multi_column_output(self, library):
        pass

    def test_is_serialisable(self, tmp_path):
        transformer = DatetimeInfoExtractor(columns=["b"], include=["timeofyear"])

        # pickle transformer
        path = tmp_path / "transformer.pkl"

        # serialise without raising error
        joblib.dump(transformer, path)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeInfoExtractor"
