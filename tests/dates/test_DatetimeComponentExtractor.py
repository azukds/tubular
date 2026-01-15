import datetime

import narwhals as nw
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseDatetimeTransformer import (
    DatetimeMixinTransformTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
)
from tubular.dates import DatetimeComponentExtractor


class TestInit(
    ColumnStrListInitTests,
):
    "tests for DatetimeComponentExtractor.__init__"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeComponentExtractor"

    @pytest.mark.parametrize(
        "incorrect_type_include",
        [2, 3.0, "invalid", ["invalid", "hour"]],
    )
    def test_error_for_bad_include_type(self, incorrect_type_include):
        """Test that an exception is raised when include variable is incorrect type."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeComponentExtractor(columns=["a"], include=incorrect_type_include)

    def test_error_when_invalid_include_option(self):
        """Test that an exception is raised when include contains incorrect values."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            DatetimeComponentExtractor(
                columns=["a"],
                include=["hour", "day", "invalid_option"],
            )


class TestTransform(
    GenericTransformTests,
    DatetimeMixinTransformTests,
):
    "tests for DatetimeComponentExtractor.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeComponentExtractor"

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_output_for_subset_of_options(self, library, from_json, lazy):
        """Test that correct df is returned after transformation."""
        # Create test data with explicit datetime values
        df = nw.from_native(d.create_date_test_df(library=library))
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

        # Initialize the transformer with the desired components to extract
        transformer = DatetimeComponentExtractor(
            columns=["b"],
            include=["hour", "day"],
        )

        if _check_if_skip_test(transformer, df, lazy, from_json):
            return

        transformer = _handle_from_json(transformer, from_json)
        transformed = transformer.transform(_convert_to_lazy(df.to_native(), lazy))

        # Define the expected output DataFrame
        expected = df.clone()
        expected = df.with_columns(
            nw.new_series(
                name="b_hour",
                values=[
                    None,
                    12.0,
                    11.0,
                    10.0,
                    18.0,
                    22.0,
                    19.0,
                    3.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
            nw.new_series(
                name="b_day",
                values=[
                    None,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    23.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
        )

        # Assert that the transformed DataFrame matches the expected output
        assert_frame_equal_dispatch(
            _collect_frame(transformed, lazy), expected.to_native()
        )

        # Test single row transformation
        df = nw.from_native(df)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row,
            )

    @pytest.mark.parametrize(
        "lazy",
        [True, False],
    )
    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_output_for_all_options(self, library, from_json, lazy):
        """Test that correct df is returned after transformation for all options, including JSON serialization."""
        # Create test data with explicit datetime values
        df = nw.from_native(d.create_date_test_df(library=library))
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

        # Initialize the transformer with all components to extract
        transformer = DatetimeComponentExtractor(
            columns=["b"],
            include=["hour", "day", "month", "year"],
        )

        if _check_if_skip_test(transformer, df, lazy, from_json):
            return

        # Handle JSON serialization and deserialization
        transformer = _handle_from_json(transformer, from_json)

        transformed = transformer.transform(_convert_to_lazy(df.to_native(), lazy))

        # Define the expected output DataFrame
        expected = df.clone()
        expected = df.with_columns(
            nw.new_series(
                name="b_hour",
                values=[
                    None,
                    12.0,
                    11.0,
                    10.0,
                    18.0,
                    22.0,
                    19.0,
                    3.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
            nw.new_series(
                name="b_day",
                values=[
                    None,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    23.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
            nw.new_series(
                name="b_month",
                values=[
                    None,
                    12.0,
                    11.0,
                    11.0,
                    9.0,
                    11.0,
                    11.0,
                    7.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
            nw.new_series(
                name="b_year",
                values=[
                    None,
                    2019.0,
                    2018.0,
                    2018.0,
                    2018.0,
                    2015.0,
                    2015.0,
                    2015.0,
                ],
                backend=backend,
                dtype=nw.Float32,
            ),
        )

        # Assert that the transformed DataFrame matches the expected output
        assert_frame_equal_dispatch(
            _collect_frame(transformed, lazy), expected.to_native()
        )

        # Test single row transformation
        df = nw.from_native(df)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(
                _convert_to_lazy(df[[i]].to_native(), lazy)
            )
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                _collect_frame(df_transformed_row, lazy),
                df_expected_row,
            )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeComponentExtractor"
