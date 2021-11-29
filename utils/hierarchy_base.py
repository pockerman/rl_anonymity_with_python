import abc
from pathlib import Path
from typing import TypeVar


HierarchyBase = TypeVar("HierarchyBase")


class HierarchyBase(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def read_from(self, filename: Path) -> HierarchyBase:
        """
        Reads the values of the hierarchy from the file
        :param filename: The file to read the values of the hierarchy
        :return: None
        """