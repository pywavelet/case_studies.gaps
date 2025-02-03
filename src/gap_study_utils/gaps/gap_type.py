from enum import Enum
from typing import List

class GapType(Enum):
    STITCH = 1
    RECTANGULAR_WINDOW = 2

    def __repr__(self):
        # name without the class name
        return self.name.split(".")[-1]

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def all_types() -> List["GapType"]:
        return [*GapType]


    @classmethod
    def parse(cls, value: str) -> "GapType":
        return GapType[value.upper()]