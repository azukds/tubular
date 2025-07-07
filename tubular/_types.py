from typing import Annotated, Union

from beartype.vale import Is

GenericKwargsType = Annotated[
    dict[str, Union[int, float, str, list[int], list[str], list[float]]],
    Is[lambda dict_arg: all(isinstance(key, str) for key in dict_arg)],
]

PositiveInt = Annotated[int, Is[lambda i: i >= 0]]

PositiveNumber = Annotated[
    Union[int, float],
    Is[lambda v: v > 0],
]

ListOfStrings = Annotated[
    list,
    Is[lambda list_arg: all(isinstance(l_value, str) for l_value in list_arg)],
]

SingleStrList = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 1],
]
