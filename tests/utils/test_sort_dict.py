from tubular._utils import _sort_dict


def test_output():
    "test output of function."

    dictionary = {"b": "bla", None: 0, "a": "b", "z": 10}

    output = _sort_dict(dictionary)

    assert output == dictionary

    assert list(output.keys()) == ["a", "b", "z", None]
