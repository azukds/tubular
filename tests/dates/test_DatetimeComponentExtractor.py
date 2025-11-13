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
from tubular.dates import DatetimeComponentExtractor


@pytest.fixture()
def hour_extractor():
    return DatetimeComponentExtractor(columns=["a"], include=["hour"])


@pytest.fixture()
def day_extractor():
    return DatetimeComponentExtractor(columns=["a"], include=["day"])


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
    DropOriginalTransformMixinTests,
):
    "tests for DatetimeComponentExtractor.transform"

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeComponentExtractor"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_single_column_output_for_all_options(self, library):
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
        transformed = transformer.transform(df.to_native())

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
        assert_frame_equal_dispatch(transformed, expected.to_native())

        # Test single row transformation
        df = nw.from_native(df)
        for i in range(len(df)):
            df_transformed_row = transformer.transform(df[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )


def test_is_serialisable(tmp_path):
    transformer = DatetimeComponentExtractor(columns=["b"], include=["hour"])

    # pickle transformer
    path = tmp_path / "transformer.pkl"

    # serialise without raising error
    joblib.dump(transformer, path)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DatetimeComponentExtractor"
