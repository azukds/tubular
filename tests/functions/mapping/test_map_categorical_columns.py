"""Tests for map_categorical_columns function.

Function is currently primarily tested indirectly through
MappingTransformer tests, but include missed cases here.
"""

import pytest

from tubular.functions.mapping import map_categorical_columns


@pytest.mark.parametrize("verbose", [True, False])
def test_warning(verbose, recwarn):
    "test warning outputted as expected."

    map_categorical_columns(
        cols=["a"],
        mappings={"a": {"a": True}},
        mappings_from_null={"a": None},
        return_dtypes={"a": "Boolean"},
        verbose=verbose,
    )

    msg = """map_categorical_columns: Note if working in pandas, it is
            not recommended to use this function to cast to Boolean, as
            result will be 'bool' type and may corrupt null values.
            This warning can be silenced by setting verbose=False."""

    if not verbose:
        assert ~any(msg in str(w.message) for w in recwarn), (
            "unexpected warning raised from map_categorical_columns"
        )

    else:
        assert any(msg in str(w.message) for w in recwarn), (
            "expected warning not raised from map_categorical_columns"
        )
