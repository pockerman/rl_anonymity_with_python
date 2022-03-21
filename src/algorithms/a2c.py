import numpy as np
from typing import TypeVar, Generic, Any, Callable
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
    n_iterations_per_episode: int = 100
    action_sampler: Callable = None
    loss_function: LossFunction = None
    batch_size: int = 0
    device: str = 'cpu'


class A2C(Generic[Optimizer]):

    @staticmethod
    def update_parameters(optimizer: Optimizer, episode_info: EpisodeInfo, *, config: A2CConfig):
        """Update the parameters

        Parameters
        ----------

        optimizer: The optimizer instance used in training
        episode_info: The episode info
        config: The training configuration

        Returns
        -------

        """

        # unroll the batch notice the flip we go in reverse
        rewards = rewards = torch.Tensor(episode_info.info["replay_buffer"].to_numpy("reward")).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(episode_info.info["logprobs"]).flip(dims=(0,)).view(-1)
        values = torch.stack(episode_info.info["values"]).flip(dims=(0,)).view(-1)

        returns = []
        ret_ = torch.Tensor([0])

        # Loop through the rewards in reverse order to generate
        # R = r i + γ * R
        for r in range(rewards.shape[0]):  # B
            ret_ = rewards[r] + config.gamma * ret_
            returns.append(ret_)

        returns = torch.stack(returns).view(-1)
        returns = F.normalize(returns, dim=0)

        # compute the actor loss.
        # Minimize the actor loss: –1 * γ t * (R – v(s t )) * π (a ⏐ s)
        actor_loss = -1 * logprobs * (returns - values.detach())  # C

        # compute the critic loss. Minimize the critic loss: (R – v) 2 .
        critic_loss = torch.pow(values - returns, 2)  # D
        loss = actor_loss.sum() + config.tau * critic_loss.sum()  # E
        loss.backward()
        optimizer.step()

    def __init__(self, config: A2CConfig, a2c_net: A2CNet):

        self.config: A2CConfig = config
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
    def _do_train(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
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

        buffer = ReplayBuffer(options["buffer_size"])

        time_step = env.reset()
        state = torch.from_numpy(time_step.observation.to_numpy()).float()

        # represent the state function values
        values = []

        # represent the probabilities under the
        # policy
        logprobs = []

        for itr in range(self.config.n_iterations_per_episode):

            # policy and critic values
            policy, value = self.a2c_net(state)

            #
            logits = policy.view(-1)

            values.append(value)

            # choose the action this should be
            action = self.config.action_sampler(logits)

            action_applied = action
            if isinstance(action, torch.Tensor):
                action_applied = env.get_action(action.item())

            # the log probabilities of the policy
            logprob_ = policy.view(-1)[action]
            logprobs.append(logprob_)

            time_step = env.step(action_applied)

            episode_score += time_step.reward
            total_distortion += time_step.info["total_distortion"]

            state = torch.from_numpy(time_step.observation.to_numpy()).float()

            buffer.add(state=state, action=action, reward=time_step.reward, next_state=time_step.observation,
                       done=time_step.done)

            if time_step.done:
                break

            episode_iterations += 1

        episode_info = EpisodeInfo(episode_score=episode_score,
                                   total_distortion=total_distortion, episode_itrs=episode_iterations,
                                   info={"replay_buffer": buffer,
                                         "logprobs": logprobs,
                                         "values": values})
        return episode_info


