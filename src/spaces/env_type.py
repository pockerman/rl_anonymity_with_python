"""Module env_type specifies an enumeration for discrete
environment type

"""
import enum


class DiscreteEnvType(enum.IntEnum):
    """Enumeration to distinguish between
    discrete environment implementations

    """

    INVALID_STATE = -1
    TOTAL_DISTORTION_STATE = 0
    MULTI_COLUMN_STATE = 1
