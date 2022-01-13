"""
Enumeration helper for quick and uniform
access of the various distance metrics
"""
import enum


class NumericDistanceType(enum.IntEnum):
    """
      Enumeration of the various distance types
    """

    # Denotes the first `TimeStep` in a sequence.
    L1 = 0
    L2 = 1
    INVALID = 2
