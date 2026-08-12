"""Tests for map_boolean_columns function.

Function is currently primarily tested indirectly through
MappingTransformer tests, but include missed cases here.
"""

import pytest

from tubular.functions.mapping import map_boolean_columns


@pytest.mark.parametrize("verbose", [True, False])
def test_warning(verbose, recwarn):
    "test warning outputted as expected."

    map_boolean_columns(
        cols=["a"],
        mappings={"a": {False: True}},
        mappings_from_null={"a": None},
        return_dtypes={"a": "Boolean"},
        verbose=verbose,
    )

    msg = """map_boolean_columns: Note if working in pandas and casting to/from Boolean,
            expressions output by this function are only intended for use on nullable
            type columns, as non-nullable types will result in use of the non-nullable
            'bool' type which may corrupt null values.
            This warning can be silenced by setting verbose=False."""

    if not verbose:
        assert ~any(msg in str(w.message) for w in recwarn), (
            "unexpected warning raised from map_boolean_columns"
        )

    else:
        assert any(msg in str(w.message) for w in recwarn), (
            "expected warning not raised from map_boolean_columns"
        )
