"""
Various methods to calculate distance between numeric vectors
"""
import numpy as np
from typing import TypeVar

from src.utils.numeric_distance_type import NumericDistanceType
from src.exceptions.exceptions import IncompatibleVectorSizesException, InvalidParamValue

Vector = TypeVar("Vector")


class NumericDistanceCalculator(object):

    def __init__(self, dist_type: NumericDistanceType) -> None:
        self.dist_type = dist_type

    def calculate(self, state1: Vector, state2: Vector) -> float:
        return _numeric_distance_calculator(state1=state1, state2=state2, dist_type=self.dist_type)


def _numeric_distance_calculator(state1: Vector, state2: Vector, dist_type: NumericDistanceType) -> float:

    if len(state1) != len(state2):
        raise IncompatibleVectorSizesException(size1=len(state1), size2=len(state2))

    if dist_type == NumericDistanceType.L1:
        return _l1_state_leakage(state1=state1, state2=state2)
    elif dist_type == NumericDistanceType.L2:
        return _l1_state_leakage(state1=state1, state2=state2)
    elif dist_type == NumericDistanceType.L2_NORMALIZED:
        return _normalized_l2_distance(state1=state1, state2=state2)

    raise InvalidParamValue(param_name="dist_type", param_value=dist_type.name)


def _normalized_l2_distance(state1: Vector, state2: Vector) -> float:
    """
    Returns the normalized L2 norm between the two vectors
    :param state1:
    :param state2:
    :return:
    """

    size = len(state1)
    dist = 0.0
    for item1, item2 in zip(state1, state2):
        dist += ((item1 - item2)**2) / size

    return np.sqrt(dist)

def _l2_state_leakage(state1: Vector, state2: Vector) -> float:
    return np.linalg.norm(state1 - state2, ord=None)

def _l1_state_leakage(state1: Vector, state2: Vector) -> float:
    return np.linalg.norm(state1 - state2, ord=1)
