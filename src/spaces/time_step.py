"""

"""

import copy
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


def copy_time_step(time_step: TimeStep, **copy_options) -> TimeStep:
    """Helper to copy partly or in whole a TimeStep namedtuple.
    If copy_options is None or empty it returns a deep copy
    of the given time step

    Parameters
    ----------
    time_step: The time step to copy
    copy_options: Members to be copied

    Returns
    -------

    An instance of the TimeStep namedtuple

    """
    if not copy_options or len(copy_options) == 0:
        return copy.deepcopy(time_step)

    observation = copy_options["observation"] if "observation" in copy_options else time_step.observation
    step_type = copy_options["step_type"] if "step_type" in copy_options else time_step.step_type
    info = copy_options["info"] if "info" in copy_options else time_step.info
    reward = copy_options["reward"] if "reward" in copy_options else time_step.reward
    discount = copy_options["discount"] if "discount" in copy_options else time_step.discount
    return TimeStep(observation=observation, step_type=step_type, info=info,
                    reward=reward, discount=discount)



