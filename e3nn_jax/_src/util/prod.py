from functools import reduce
from typing import List, Union


def prod(list_of_numbers: List[Union[int, float]]) -> Union[int, float]:
    """Product of a list of numbers."""
    return reduce(lambda x, y: x * y, list_of_numbers, 1)
