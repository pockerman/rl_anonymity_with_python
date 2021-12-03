import numpy as np
import textdistance
import enum
from src.exceptions import Error


class DistanceType(enum.IntEnum):
    """
      Defines the status of a `TimeStep` within a sequence.
      """

    # Denotes the first `TimeStep` in a sequence.
    COSINE = 0


class TextDistanceCalculator(object):
    """
    Wrapper class for text distance calculation
    """

    DISTANCE_TYPES = [DistanceType.COSINE, ]

    @staticmethod
    def build_calculator(dist_type: DistanceType):

        if dist_type not in TextDistanceCalculator.DISTANCE_TYPES:
            raise Error("Distance type '{0}' is invalid".format(str(dist_type)))

        if dist_type == DistanceType.COSINE:
            return textdistance.Cosine()

    def __init__(self, dist_type):

        if dist_type not in TextDistanceCalculator.DISTANCE_TYPES:
            raise Error("Distance type '{0}' is invalid".format(dist_type))

        self._dist_type = dist_type

    def calculate(self, txt1, txt2, **options):

        # build a calculator
        calculator = TextDistanceCalculator.build_calculator(dist_type=self._dist_type)

        set_options = getattr(calculator, "set_options", None)

        if set_options is not None:
            calculator.set_options(**options)

        return calculator.similarity(txt1, txt2)










