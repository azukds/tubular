from typing import Annotated, Union

from beartype.vale import Is

# needed as by default beartype will just randomly sample to type check elements
# and we want consistency
ListOfStrs = Annotated[
    list,
    Is[lambda list_arg: all(isinstance(l_value, str) for l_value in list_arg)],
]

ListOfOneStr = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 1],
]

ListOfTwoStrs = Annotated[
    list[str],
    Is[lambda list_arg: len(list_arg) == 2],
]

PositiveNumber = Annotated[
    Union[int, float],
    Is[lambda v: v > 0],
]

PositiveInt = Annotated[int, Is[lambda i: i >= 0]]

GenericKwargs = Annotated[
    dict,
    Is[
        lambda d: all(
            isinstance(key, str) and isinstance(value, (str, int, float))
            for key, value in d.items()
        )
    ],
]
