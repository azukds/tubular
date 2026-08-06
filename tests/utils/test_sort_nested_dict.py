from tubular._utils import _sort_nested_dict


def test_output():
    "test output of function."

    dictionary = {
        "b": {"b": 1, "a": 2, None: 3, "z": 4},
        None: {"z": "a", "t": "b"},
        "c": {},
    }

    output = _sort_nested_dict(dictionary)

    assert output == dictionary

    assert list(output.keys()) == ["b", "c", None]
    assert list(output["b"].keys()) == ["a", "b", "z", None]
    assert list(output[None].keys()) == ["t", "z"]
    assert list(output["c"].keys()) == []
