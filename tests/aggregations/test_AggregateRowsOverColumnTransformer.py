"""Unit tests for AggregateRowsOverColumnTransformer."""

import pytest

import dataframes as d
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.aggregations import AggregateRowsOverColumnTransformer


class TestTransform:
    """Tests for transform method."""

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_basic_aggregation(self, library):
        """Test basic aggregation functionality."""
        df = dataframe_init_dispatch(library, {"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        transformer = AggregateRowsOverColumnTransformer(
            columns="value",
            aggregations=["mean"],
            key="group",
        )

        df_transformed = transformer.transform(df)

        expected = dataframe_init_dispatch(
            library,
            {
                "group": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
                "value_mean": [1.5, 1.5, 3.5, 3.5],
            },
        )

        assert_frame_equal_dispatch(library, df_transformed, expected)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_multiple_aggregations(self, library):
        """Test multiple aggregations."""
        df = dataframe_init_dispatch(library, {"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        transformer = AggregateRowsOverColumnTransformer(
            columns="value",
            aggregations=["min", "max"],
            key="group",
        )

        df_transformed = transformer.transform(df)

        expected = dataframe_init_dispatch(
            library,
            {
                "group": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
                "value_min": [1, 1, 3, 3],
                "value_max": [2, 2, 4, 4],
            },
        )

        assert_frame_equal_dispatch(library, df_transformed, expected)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_drop_original(self, library):
        """Test dropping original columns."""
        df = dataframe_init_dispatch(library, {"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        transformer = AggregateRowsOverColumnTransformer(
            columns="value",
            aggregations=["mean"],
            key="group",
            drop_original=True,
        )

        df_transformed = transformer.transform(df)

        expected = dataframe_init_dispatch(
            library,
            {
                "group": ["A", "A", "B", "B"],
                "value_mean": [1.5, 1.5, 3.5, 3.5],
            },
        )

        assert_frame_equal_dispatch(library, df_transformed, expected)


class TestToJson:
    """Tests for to_json method."""

    def test_to_json_returns_correct_dict(self):
        """Test that to_json is working as expected."""
        transformer = AggregateRowsOverColumnTransformer(
            columns="a",
            aggregations=["min", "max"],
            key="b",
            drop_original=False,
        )

        actual = transformer.to_json()

        # check tubular_version is present and a string, then remove
        assert isinstance(
            actual["tubular_version"],
            str,
        ), "expected tubular version to be captured as str in to_json"
        del actual["tubular_version"]

        expected = {
            "classname": "AggregateRowsOverColumnTransformer",
            "init": {
                "columns": ["a"],
                "copy": False,
                "verbose": False,
                "return_native": True,
                "aggregations": ["min", "max"],
                "drop_original": False,
                "key": "b",
            },
            "fit": {},
        }

        assert actual == expected, "to_json does not return the expected dictionary"

    def test_to_json_with_drop_original_true(self):
        """Test that to_json correctly captures drop_original=True."""
        transformer = AggregateRowsOverColumnTransformer(
            columns=["a", "b"],
            aggregations=["sum"],
            key="group",
            drop_original=True,
        )

        actual = transformer.to_json()

        # check tubular_version is present and a string, then remove
        assert isinstance(
            actual["tubular_version"],
            str,
        ), "expected tubular version to be captured as str in to_json"
        del actual["tubular_version"]

        expected = {
            "classname": "AggregateRowsOverColumnTransformer",
            "init": {
                "columns": ["a", "b"],
                "copy": False,
                "verbose": False,
                "return_native": True,
                "aggregations": ["sum"],
                "drop_original": True,
                "key": "group",
            },
            "fit": {},
        }

        assert actual == expected, "to_json does not return the expected dictionary"
