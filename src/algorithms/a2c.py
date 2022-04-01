import numpy as np
from typing import TypeVar, Generic, Any, Callable, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.utils import INFO
from src.utils.episode_info import EpisodeInfo
from src.utils.function_wraps import time_func_wrapper
from src.utils.replay_buffer import ReplayBuffer
from src.spaces.time_step import VectorTimeStep
from src.maths.pytorch_optimizer_config import PyTorchOptimizerConfig
from src.maths.pytorch_optimizer_builder import pytorch_optimizer_builder
from src.maths.loss_functions import mse

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
LossFunction = TypeVar("LossFunction")
State = TypeVar("State")
Action = TypeVar("Action")
TimeStep = TypeVar("TimeStep")
Criteria = TypeVar('Criteria')


def create_discounts_array(end: int, base: float, start=0, endpoint=False):
    """

    Parameters
    ----------
    end
    base
    start
    endpoint

    Returns
    -------

    """
    return np.logspace(start, end, num=end, base=base, endpoint=endpoint)


def calculate_discounted_returns(rewards: List[float], discounts: List[float], n_workers: int = 1) -> np.array:
    """Calculate the discounted returns from the episode rewards

    Parameters
    ----------
    rewards: The list of rewards
    gamma: The discount factor
    n_workers: The number of workers

    Returns
    -------

    """

    # T
    total_time = len(rewards)

    # Return numbers spaced evenly on a log scale.
    # In linear space, the sequence starts at base ** start
    # (base to the power of start) and ends with base ** stop (see endpoint below).
    #discounts = np.logspace(0, total_time, num=total_time, base=gamma, endpoint=False)

    # The return is the sum of discounted rewards from step until the
    # final step T
    returns = np.array([[np.sum(discounts[: total_time - t] * rewards[t:, w]) for t in range(total_time)] for w in range(n_workers)])
    return returns


def generalized_advantage_estimate(rewards: List[float],
                                   values: List[float],
                                   gamma: float, tau: float,
                                   n_workers: int) -> np.array:
    """Computes an estimate of the advantage funcion

    Parameters
    ----------
    rewards
    values
    gamma
    tau
    n_workers: The number of workers

    Returns
    -------

    """

    # T
    total_time = len(rewards)

    # (gamma*tau)^t
    tau_discounts = np.logspace(0, total_time - 1, num=total_time-1,
                                base=gamma*tau, endpoint=False)

    # create TD errors: R_t + gamma*V_{t+1} - V_t for t=0 to T
    advantages = rewards[:-1] + gamma * values[1:] - values[: -1]

    # create the GAES by multiplying the tau discounts times the TD errors
    gaes = np.array([[np.sum(tau_discounts[: total_time - 1 - t] * advantages[t:]) for t in range(total_time)] for w in range(n_workers)])
    return gaes


@dataclass(init=True, repr=True)
class A2CConfig(object):
    """Configuration for A2C algorithm

    """

    gamma: float = 0.99
    tau: float = 1.2
    beta: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    n_iterations_per_episode: int = 100
    n_workers: int = 1
    action_sampler: Callable = None
    value_function: LossFunction = None
    policy_loss: LossFunction = None
    batch_size: int = 0
    device: str = 'cpu'
    a2cnet: nn.Module = None
    save_model_path: Path = None
    optimizer_config: PyTorchOptimizerConfig = None


@dataclass(init=True, repr=True)
class _InteractionResult(object):
    logpas: float
    entropies = None
    rewards = None
    values = None


@dataclass(init=True, repr=True)
class _FullPassResult(object):
    logprobs: torch.Tensor
    values: torch.Tensor
    actions: torch.Tensor
    entropies: torch.Tensor


class A2C(Generic[Optimizer]):

    @staticmethod
    def default_action_sampler(logits: torch.Tensor) -> torch.distributions.Distribution:

        action_dist = torch.distributions.Categorical(logits=logits)
        return action_dist

    @classmethod
    def from_path(cls, config: A2CConfig, path: Path):
        """Load the A2C model parameters from the given path

        Parameters
        ----------
        config: The configuration of the algorithm
        path: The path to load the parameters

        Returns
        -------

        An instance of A2C class
        """
        a2c = A2C(config)
        a2c.config.a2cnet.load_state_dict(torch.load(path))
        a2c.set_test_mode()
        return a2c

    def __init__(self, config: A2CConfig):

        self.config: A2CConfig = config
        self.optimizer: Optimizer = None
        self.name = "A2C"

    @property
    def a2c_net(self) -> nn.Module:
        return self.config.a2cnet

    def __call__(self, x: torch.Tensor):
        return self.a2c_net(x)

    def parameters(self) -> Any:
        """The parameters of the underlying model

        Returns
        -------

        """
        return self.a2c_net.parameters()

    def set_train_mode(self) -> None:
        """Set the model to a training mode

        Returns
        -------
        None
        """
        self.a2c_net.train()

    def set_test_mode(self) -> None:
        """Set the model to a testing mode

        Returns
        -------
        None
        """
        self.a2c_net.eval()

    def save_model(self, path: Path) -> None:
        """Save the model on a file at the given path

        Parameters
        ----------
        path: The path to save the model

        Returns
        -------

        None
        """
        torch.save(self.a2c_net.state_dict(), Path(str(path) + "/" + self.name + ".pt"))

    def play(self, env: Env, criteria: Criteria):
        """Play the agent on the environment

        Parameters
        ----------
        env: The environment to test/play the agent
        criteria: The criteria to stop the game

        Returns
        -------

        """

        time_step = env.reset()

        while criteria.continue_itrs():
            state = time_step.observation.to_ndarray()
            state = torch.from_numpy(state).float()
            logits, values = self(state)

            # select action
            action = None
            time_step = env.step(action)

            if time_step.done:
                time_step = env.reset()

    @time_func_wrapper(show_time=False)
    def optimize_model(self, logpas, entropies, values, rewards, n_workers: int) -> None:

        print("{0} optimizing model={1}".format(INFO, self.name))
        discounts = create_discounts_array(end=len(rewards), base=self.config.gamma, start=0, endpoint=False)

        # get the discounted returns
        discounted_returns = calculate_discounted_returns(rewards.detach().numpy(), discounts, n_workers=n_workers)

        # get the gaes
        gaes = generalized_advantage_estimate(rewards=rewards.detach().numpy(), gamma=self.config.gamma,
                                              values=values.detach().numpy(),
                                              tau=self.config.tau, n_workers=self.config.n_workers)

        # discounted gaes
        discounted_gaes = discounts[:-1] * gaes

        # the loss function for the critic network
        value_loss_function = mse(returns=discounted_returns, values=values)
        policy_loss = - (discounted_gaes * logpas).mean()

        # compute a total loss function to minimize
        if self.config.beta is not None:

            # add entropy loss
            entropy_loss = -entropies.mean()

            loss = self.config.policy_loss_weight * policy_loss + \
                self.config.value_loss_weight * value_loss_function + \
                self.config.beta * entropy_loss
        else:
            loss = self.config.policy_loss_weight * policy_loss + \
                   self.config.value_loss_weight * value_loss_function

        self.optimizer.zero_grad()
        loss.backward()

        # clip the grad if needed
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.config.max_grad_norm)
        self.optimizer.step()

    def actions_before_training_begins(self, env: Env, **options) -> None:

        # build the optimizer we need in order to train the model
        self.optimizer = pytorch_optimizer_builder(opt_type=self.config.optimizer_config.optimizer_type,
                                                   model_params=self.parameters(),
                                                   **self.config.optimizer_config.as_dict())

        if self.config.action_sampler is None:
            self.config.action_sampler = A2C.default_action_sampler

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        self.set_train_mode()
        self.optimizer.zero_grad()

    def actions_after_training(self) -> None:
        """Any actions the agent needs to perform after training

        Returns
        -------

        None
        """

        if self.config.save_model_path is not None:
            self.save_model(path=self.config.save_model_path)

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Actions the agent applis after the episode ends

        Parameters
        ----------
        env
        episode_idx
        options

        Returns
        -------

        """

        episode_info: EpisodeInfo = options["episode_info"]

        buffer: ReplayBuffer = episode_info.info["buffer"]

        self.optimize_model(rewards=buffer.get_item_as_torch_tensor("reward"),
                            logpas=buffer.get_torch__tensor_info_item_as_torch_tensor("logprobs"),
                            values=buffer.get_torch__tensor_info_item_as_torch_tensor("values"),
                            entropies=buffer.get_torch__tensor_info_item_as_torch_tensor("entropies"),
                            n_workers=env.n_workers)

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

        time_step: VectorTimeStep = env.reset()
        states = time_step.stack_observations()
        #torch.from_numpy(time_step.observation.to_numpy()).float()

        buffer = ReplayBuffer(buffer_size=self.config.n_iterations_per_episode)

        for itr in range(self.config.n_iterations_per_episode):

            full_pass: _FullPassResult = self._network_pass(states)
            time_step: VectorTimeStep = env.step(full_pass.actions)
            next_states = time_step.stack_observations()

            buffer.add(state=states, next_state=next_states,
                       reward=time_step.stack_rewards(),
                       action=full_pass.actions,
                       done=time_step.stack_dones(),
                       info={"values": full_pass.values,
                             "entropies": full_pass.entropies,
                             "logprobs": full_pass.logprobs})

            states = next_states

            if time_step.done:
                break

            episode_iterations += 1

        episode_info = EpisodeInfo(episode_score=episode_score,
                                   total_distortion=total_distortion, episode_itrs=episode_iterations,
                                   info={"buffer": buffer})
        return episode_info

    def _network_pass(self, state) -> _FullPassResult:

        if not isinstance(state, torch.Tensor):
            torch_state = torch.Tensor(state)
        else:
            torch_state = state

        # policy and critic values. The policy
        # values are assumed raw
        logits, values = self.a2c_net(torch_state)

        # log_softmax may not sum up to one
        # and can be negative as well
        # get the logprobs for all batches?
        logprobs = F.log_softmax(logits.view(-1), dim=0)

        # choose the action. Typically this will be Categorical
        # but we leave it open for the application
        # We don't call logits.view(-1) so that we get
        # as many actions as in the logits rows.
        # Each logit row is expected to corrspond to an
        # environment worker
        action_sampler_dist = self.config.action_sampler(logits)
        actions = action_sampler_dist.sample()
        entropies = action_sampler_dist.entropy().unsqueeze(-1)

        full_pass_result = _FullPassResult(logprobs=logprobs, actions=actions,
                                           values=values, entropies=entropies)

        return full_pass_result
