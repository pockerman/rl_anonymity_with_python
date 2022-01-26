"""
Utilities for dataset distortion calculation
"""
import enum
from typing import TypeVar
from src.utils.numeric_distance_type import NumericDistanceType
from src.utils.numeric_distance_calculator import NumericDistanceCalculator
from src.utils.string_distance_calculator import StringDistanceType, TextDistanceCalculator
from src.exceptions.exceptions import InvalidParamValue

Vector = TypeVar('Vector')


class DistortionCalculationType(enum.IntEnum):
    """

    """

    INVALID = -1
    SUM = 0
    AVG = 1


class DistortionCalculator(object):

    def __init__(self, numeric_column_distortion_metric_type: NumericDistanceType,
                 string_column_distortion_metric_type: StringDistanceType,
                 dataset_distortion_type: DistortionCalculationType):
        self.numeric_column_distortion_metric_type = numeric_column_distortion_metric_type
        self.string_column_distortion_metric_type = string_column_distortion_metric_type
        self.dataset_distortion_type = dataset_distortion_type

    def calculate(self, vec1: Vector, vec2: Vector, datatype: str) -> float:

        if datatype == 'str':
            return TextDistanceCalculator(dist_type=self.string_column_distortion_metric_type).calculate(txt1=vec1,
                                                                                                         txt2=vec2)
        elif datatype == 'float' or datatype == 'int':
            return NumericDistanceCalculator(dist_type=self.numeric_column_distortion_metric_type).calculate(state1=vec1,
                                                                                                             state2=vec2)
        raise InvalidParamValue(param_name='datatype', param_value=datatype)

    def total_distortion(self, distortions: Vector) -> float:

        if self.dataset_distortion_type == DistortionCalculationType.SUM:
            return float(sum(distortions))
        elif self.dataset_distortion_type == DistortionCalculationType.AVG:
            return float(sum(distortions) / len(distortions))

        raise InvalidParamValue(param_name='dataset_distortion_type', param_value=self.dataset_distortion_type.name)

