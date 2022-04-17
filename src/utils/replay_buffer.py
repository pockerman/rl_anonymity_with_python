import random
from typing import Any, List
from collections import namedtuple, deque
import numpy as np
import torch

from src.exceptions.exceptions import InvalidParamValue

ExperienceTuple = namedtuple("ExperienceTuple", field_names=["state", "action",
                                                             "reward", "next_state", "done", "info"])


class ReplayBuffer(object):
    """The ReplayBuffer class.
    Models a fixed size replay buffer.
    The buffer is represented by using a deque from Python’s built-in collections library.
    This is basically a list that we can set a maximum size.  If we try to add a new element whilst the list
    is already full, it will remove the first item in the list and add the new item to the end of the list.
    Hence new experiences  replace the oldest experiences.
    The experiences themselves are tuples of (state1, reward, action, state2, done) that we append to the replay deque
    and they are represented via the named tuple ExperienceTuple
    """

    TUPLE_NAMES = ["state", "action", "reward", "next_state", "done", "info"]

    def __init__(self, buffer_size: int):
        """Constructor

        Parameters
        ----------

        buffer_size: The maximum capacity of the buffer

        """

        self.capacity: int = buffer_size
        self._memory = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        """ Return the current size of the internal memory.

        Returns
        -------

        """
        return len(self._memory)

    def __getitem__(self, name_attr: str) -> List:
        """Return the full batch of the name_attr attribute

        Parameters
        ----------
        name_attr: The name of the attribute to collect the
        batch values

        Returns
        -------

        A list
        """

        if name_attr not in ReplayBuffer.TUPLE_NAMES:
            raise InvalidParamValue(param_name=name_attr, param_value=name_attr)

        batch = []
        for item in self._memory:

            if name_attr == "action":
                batch.append(item.action)
            elif name_attr == "state":
                batch.append(item.state)
            elif name_attr == "next_state":
                batch.append(item.next_state)
            elif name_attr == "reward":
                batch.append(item.reward)
            elif name_attr == "done":
                batch.append(item.done)
            elif name_attr == "info":
                batch.append(item.info)

        return batch

    def to_numpy(self, name_attr: str) -> np.array:
        return np.array(self[name_attr])

    def get_item_as_torch_tensor(self, name_attr: str) -> torch.Tensor:
        """ Returns a torch.Tensor representation of the
        the named item

        Parameters
        ----------

        name_attr: The name of the attribute

        Returns
        -------

        An instance of  torch.Tensor
        """

        items = self[name_attr]

        # convert to np.array to avoid pytorch warning
        return torch.Tensor(np.array(items))

    def get_torch__tensor_info_item_as_torch_tensor(self, name_attr: str) -> torch. Tensor:

        vals = []
        for item in self._memory:
            info = item.info

            if name_attr in info:
                vals.append(info[name_attr])

        return torch.stack(vals)

    def add(self, state: Any, action: Any, reward: Any,
            next_state: Any, done: Any, info: dict = {}) -> None:
        """Add a new experience tuple in the buffer

        Parameters
        ----------

        state: The current state
        action: The action taken
        reward: The reward observed
        next_state: The next state observed
        done: Whether the episode is done
        info: Any other info needed

        Returns
        -------
        None

        """

        e = ExperienceTuple(state, action, reward, next_state, done, info)
        self._memory.append(e)

    def sample(self, batch_size: int) -> List[ExperienceTuple]:
        """Randomly sample a batch of experiences from memory.

        Parameters
        ----------

        batch_size: The batch size we want to sample

        Returns
        -------

        A list of ExperienceTuple
        """

        return random.sample(self._memory, k=batch_size)



    def reinitialize(self) -> None:
        """Reinitialize the internal buffer

        Returns
        -------

        None

        """

        self._memory = deque(maxlen=self.capacity)



