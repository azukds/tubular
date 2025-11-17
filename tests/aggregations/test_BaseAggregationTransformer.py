"""Unit tests for BaseAggregationTransformer."""

import pytest

from tubular.aggregations import AggregateRowsOverColumnTransformer, BaseAggregationTransformer


class TestBaseAggregationTransformer:
    """Tests for BaseAggregationTransformer."""

    def test_init_raises_error_if_called_directly(self):
        """Test that BaseAggregationTransformer raises error if instantiated directly."""
        with pytest.raises(NotImplementedError, match="BaseAggregationTransformer is an abstract class"):
            BaseAggregationTransformer(columns="a", aggregations=["min"])

    def test_to_json_returns_correct_dict(self):
        """Test that to_json is working as expected."""
        # Use AggregateRowsOverColumnTransformer as concrete implementation
        transformer = AggregateRowsOverColumnTransformer(
            columns="a",
            aggregations=["min", "max"],
            key="b",
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
                "columns": ["a"],
                "copy": False,
                "verbose": False,
                "return_native": True,
                "aggregations": ["min", "max"],
                "drop_original": True,
                "key": "b",
            },
            "fit": {},
        }

        assert actual == expected, "to_json does not return the expected dictionary"
