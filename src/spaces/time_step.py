"""

"""

import enum
from typing import NamedTuple, Generic, Optional, TypeVar

_Reward = TypeVar('_Reward')
_Discount = TypeVar('_Discount')
_Observation = TypeVar('_Observation')


class StepType(enum.IntEnum):
    """
      Defines the status of a `TimeStep` within a sequence.
      """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0

    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1

    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST


class TimeStep(NamedTuple, Generic[_Reward, _Discount, _Observation]):
    step_type: StepType
    info: dict
    reward: Optional[_Reward]
    discount: Optional[_Discount]
    observation: _Observation

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    @property
    def done(self) -> bool:
        return self.last()