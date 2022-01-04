import numpy as np
from typing import TypeVar, Generic
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class A2CConfig(object):
    """
    Configuration for A2C algorithm
    """

    def __init__(self):
        self.gamma: float = 0.99
        self.tau: float = 1.2
        self.n_workers: int = 1
        self.n_iterations_per_episode: int = 100
        self.optimizer: Optimizer = None
        self.loss_function: LossFunction = None


class A2C(Generic[Optimizer]):

    def __init__(self, config: A2CConfig, a2c_net: A2CNet):

        self.gamma = config.gamma
        self.tau = config.tau
        self.n_workers = config.n_workers
        self.n_iterations_per_episode = config.n_iterations_per_episode
        self.optimizer = config.optimizer
        self.loss_function = config.loss_function
        self.a2c_net = a2c_net
        self.rewards = []
        self.name = "A2C"

    def _optimize_model(self):
        pass

    def select_action(self, env: Env, observation: State) -> Action:
        """
        Select an action
        :param env: The environment over which the agent is trained
        :param observation: The current observation of the environment
        :return: Returns an Action type
        """
        return env.sample_action()

    def update(self):
        pass

    def train(self, env: Env) -> None:

        # reset the environment and obtain the
        # the time step
        time_step: TimeStep = env.reset()

        observation = time_step.observation

        # learn over the episode
        for iteration in range(1, self.n_iterations_per_episode + 1):

            # select an action
            action = self.select_action(env=env, observation=observation)

            # step in the environment according
            # to the selected action
            next_time_step = env.step(action=action)

            # we reached the end of the episode
            if next_time_step.last():
                break

            next_state = next_time_step.observation
            policy_val, v_val = self.a2c_net.forward(x=next_state)
            self._optimize_model()

