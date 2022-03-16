import numpy as np
from typing import TypeVar, Generic, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.utils.experience_buffer import unpack_batch
from src.utils.episode_info import EpisodeInfo
from src.utils.function_wraps import time_func_wrapper

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
LossFunction = TypeVar("LossFunction")
State = TypeVar("State")
Action = TypeVar("Action")
TimeStep = TypeVar("TimeStep")


class A2CNetBase(nn.Module):
    """
    Base class for A2C networks
    """

    def __init__(self, architecture):
        super(A2CNetBase, self).__init__()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)


class A2CNet(nn.Module):

    def __init__(self, common_net: A2CNetBase, policy_net: A2CNetBase, value_net: A2CNetBase):
        super(A2CNet, self).__init__()
        self.common_net = common_net
        self.policy_net = policy_net
        self.value_net = value_net

    def forward(self, x):
        x = self.common_net(x)

        pol_out = self.policy_net(x)
        val_out = self.value_net(x)
        return pol_out, val_out


@dataclass(init=True, repr=True)
class A2CConfig(object):
    """Configuration for A2C algorithm

    """

    gamma: float = 0.99
    tau: float = 1.2
    n_workers: int = 1
    n_iterations_per_episode: int = 100
    optimizer: Optimizer = None
    loss_function: LossFunction = None
    batch_size: int = 0
    device: str = 'cpu'


class A2C(Generic[Optimizer]):

    def __init__(self, config: A2CConfig, a2c_net: A2CNet):

        self.gamma = config.gamma
        self.tau = config.tau
        self.n_workers = config.n_workers
        self.n_iterations_per_episode = config.n_iterations_per_episode
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.device = config.device
        self.loss_function = config.loss_function
        self.a2c_net = a2c_net
        self.rewards = []
        self.memory = []
        self.name = "A2C"

    def share_memory(self) -> None:
        """Instruct the underlying network to
        set up what is needed to share memory

        Returns
        -------

        None
        """
        self.a2c_net.share_memory()

    def parameters(self) -> Any:
        return self.a2c_net.parameters()

    def select_action(self, env: Env, observation: State) -> Action:
        """
        Select an action
        :param env: The environment over which the agent is trained
        :param observation: The current observation of the environment
        :return: Returns an Action type
        """
        return env.sample_action()

    def update_policy_network(self):
        """
        Update the policy network
        :return:
        """
        pass

    def calculate_loss(self):
        """
        Calculate the loss
        :return:
        """
        pass

    def accummulate_batch(self):
        """
        Accumulate the memory items
        :return:
        """
        pass

    """
    def train(self, env: Env) -> None:
        
        Train the agent on the given environment
        :param env:
        :return:
        

        # reset the environment and obtain the
        # the time step
        time_step: TimeStep = env.reset()

        observation = time_step.observation

        # the batch to process
        batch = []

        # learn over the episode
        for iteration in range(1, self.n_iterations_per_episode + 1):

            # select an action
            action = self.select_action(env=env, observation=observation)

            # step in the environment according
            # to the selected action
            next_time_step = env.step(action=action)

            batch.append(next_time_step.observation)

            if len(batch) < self.batch_size:
                continue

            # unpack the batch in order to process it
            states_v, actions_t, vals_ref = unpack_batch(batch=batch, net=self.a2c_net, device=self.device)
            batch.clear()

            self.optimizer.zero_grad()
            # we reached the end of the episode
            #if next_time_step.last():
            #    break

            #next_state = next_time_step.observation
            policy_val, v_val = self.a2c_net.forward(x=states_v)

            self.optimizer.zero_grad()

            # claculate loss
            loss = self.calculate_loss()
            loss.backward()
            self.optimizer.step()
    """

    def on_episode(self, env: Env, episode_idx: int,  **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        episode_info, total_time = self._do_train(env, episode_idx, **options)
        episode_info.total_execution_time = total_time
        return episode_info

    @time_func_wrapper(show_time=False)
    def _do_train(self, env: Env, episode_idx: int, **option) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        # episode score
        episode_score = 0
        episode_iterations = 0
        total_distortion = 0

        episode_info = EpisodeInfo(episode_score=episode_score, total_distortion=total_distortion, episode_itrs=episode_iterations)
        return episode_info

    def actions_before_training(self, env: Env, **options) -> None:
        """Any actions before training begins

        Parameters
        ----------

        env: The environment that training occurs
        options: Any options passed by the client code

        Returns
        -------
        None
        """

        """
        if not isinstance(self.config.policy, WithQTableMixinBase):
            raise InvalidParamValue(param_name="policy", param_value=str(self.config.policy))

        for state in range(1, env.n_states):
            for action in range(env.n_actions):
                self.q_table[state, action] = 0.0
        """

    def actions_before_episode_begins(self, env: Env, episode_idx, **options) -> None:
        """Execute any actions the algorithm needs before
        the episode ends

        Parameters
        ----------

        env: The environment that training occurs
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------

        None
        """

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs after
        the episode ends

        Parameters
        ----------
        env: The environment that training occurs
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------
        None

        """
        #self.config.policy.actions_after_episode(episode_idx)

