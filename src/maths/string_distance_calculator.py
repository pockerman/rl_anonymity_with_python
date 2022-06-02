import numpy as np
import textdistance
import enum
from src.exceptions.exceptions import Error


class StringDistanceType(enum.IntEnum):
    """
    Enumeration for distinguishing among different string distance metrics.
    """

    INVALID = -1
    COSINE = 0
    COSINE_NORMALIZE = 1
    HAMMING = 2
    HAMMING_NORMALIZE = 3


class TextDistanceCalculator(object):
    """
    Wrapper class for text distance calculation
    """

    DISTANCE_TYPES = [StringDistanceType.COSINE, StringDistanceType.HAMMING,
                      StringDistanceType.COSINE_NORMALIZE, StringDistanceType.HAMMING_NORMALIZE]

    NORMALIZED_DISTANCE_TYPES = [StringDistanceType.COSINE_NORMALIZE,
                                 StringDistanceType.HAMMING_NORMALIZE]

    @staticmethod
    def build_calculator(dist_type: StringDistanceType):

        if dist_type not in TextDistanceCalculator.DISTANCE_TYPES:
            raise Error("Distance type '{0}' is invalid".format(str(dist_type)))

        if dist_type == StringDistanceType.COSINE or dist_type == StringDistanceType.COSINE_NORMALIZE:
            return textdistance.Cosine()
        elif dist_type == StringDistanceType.HAMMING or dist_type == StringDistanceType.HAMMING_NORMALIZE:
            return textdistance.Hamming()

    def __init__(self, dist_type):

        if dist_type not in TextDistanceCalculator.DISTANCE_TYPES:
            raise Error("Distance type '{0}' is invalid".format(dist_type))

        self._dist_type = dist_type

    @property
    def distance_type(self) -> StringDistanceType:
        return self._dist_type

    def calculate(self, txt1, txt2, **options) -> float:
        """Calculate the distance between the two strings

        Parameters
        ----------
        txt1: The first string
        txt2: The second string
        options: Any application options

        Returns
        -------

        A floating point number representing the distance between the
        two strings
        """

        # build a calculator
        calculator = TextDistanceCalculator.build_calculator(dist_type=self._dist_type)

        set_options = getattr(calculator, "set_options", None)

        if set_options is not None:
            calculator.set_options(**options)

        if self._dist_type in TextDistanceCalculator.NORMALIZED_DISTANCE_TYPES:
            return calculator.normalized_distance(txt1, txt2)

        return calculator.distance(txt1, txt2)










