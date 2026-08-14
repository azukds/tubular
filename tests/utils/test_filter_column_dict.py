import pytest

from tubular._utils import _filter_column_dict


@pytest.mark.parametrize(
    ("input_dict", "input_columns", "expected"),
    [({"a": 1, "b": 2}, ["a"], {"a": 1}), ({"bla": "a"}, ["blabla"], {})],
)
def test_output(input_dict, input_columns, expected):
    "test basic output cases."

    assert expected == _filter_column_dict(input_dict, input_columns)
