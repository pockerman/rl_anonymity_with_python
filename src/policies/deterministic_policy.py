import numpy as np
from typing import TypeVar

PolicyBase = TypeVar('PolicyBase')


class DeterministicAdaptorPolicy(object):

    """
    Update a policy by choosing the best action
    """

    def __init__(self) -> None:
        super(DeterministicAdaptorPolicy, self).__init__()

    def __call__(self, policy: PolicyBase, *args, **kwargs) -> PolicyBase:
        s: int = kwargs["s"]
        state_actions: np.ndarray = kwargs["state_actions"]
        action = np.argmax(state_actions)
        policy[s][action] = 1.0
        return policy