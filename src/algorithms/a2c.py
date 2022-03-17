import numpy as np
from typing import TypeVar, Generic, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


from src.utils.experience_buffer import unpack_batch
from src.utils.episode_info import EpisodeInfo
from src.utils.function_wraps import time_func_wrapper
from src.utils.replay_buffer import ReplayBuffer

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
LossFunction = TypeVar("LossFunction")
State = TypeVar("State")
Action = TypeVar("Action")
TimeStep = TypeVar("TimeStep")




class A2CNetBase(nn.Module):
    """Base class for A2C networks

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

    @staticmethod
    def update_parameters(optimizer: Optimizer, episode_info: EpisodeInfo, *, gamma: float):
        """Update the parameters

        Parameters
        ----------
        optimizer
        episode_info
        gamma

        Returns
        -------

        """

        # unroll the batch
        rewards = episode_info.info["replay_buffer"]["reward"]
        returns = []

        for r in range(rewards.shape[0]):  # B
            ret_ = rewards[r] + gamma * ret_
            returns.append(ret_)

        returns = torch.stack(returns).view(-1)
        returns = F.normalize(returns, dim=0)

        actor_loss = -1 * logprobs * (returns - values.detach())  # C
        critic_loss = torch.pow(values - returns, 2)  # D
        loss = actor_loss.sum() + clc * critic_loss.sum()  # E
        loss.backward()
        optimizer.step()

    def __init__(self, config: A2CConfig, a2c_net: A2CNet):

        self.config: A2CConfig = config

        self.tau = config.tau

        self.a2c_net = a2c_net
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
        """Train the algorithm on the episode. In fact this method simply
        plays the environment to collect batches

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

        time_step = env.reset()
        state = torch.from_numpy(time_step.observation()).float()

        values = []
        for itr in range(self.config.n_iterations_per_episode):

            # policy and critic values
            policy, value = self.a2c_net(state)

            values.append(value)

            # choose the action
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            time_step = env.step(action)

            if time_step.done:
                break

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

